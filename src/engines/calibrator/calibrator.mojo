from engines.calibrator.objective import ObjectiveFunction
from engines.fpe.heston_params import HestonParams
from numerics.optim.lm import LevenbergMarquardt, ResidualCallable, JacobianCallable
from numerics.utils import abs_f64, max_f64, min_f64, zeros
from std.algorithm import parallelize


def _params_to_vec(p: HestonParams) -> List[Float64]:
    return [p.kappa, p.theta, p.sigma, p.rho, p.V0]


def _vec_to_params(x: List[Float64], base: HestonParams) -> HestonParams:
    return HestonParams(
        kappa=max_f64(x[0], 1e-4),
        theta=max_f64(x[1], 1e-5),
        sigma=max_f64(x[2], 1e-4),
        rho=min_f64(0.999, max_f64(-0.999, x[3])),
        r=base.r,
        T=base.T,
        S0=base.S0,
        V0=max_f64(x[4], 1e-6),
        S_min=base.S_min,
        S_max=base.S_max,
        V_min=base.V_min,
        V_max=base.V_max,
    )


struct CalibratorResidual[B: Int](ResidualCallable):
    """Residual callable for LM optimizer."""
    var obj: ObjectiveFunction[Self.B]
    var base: HestonParams

    def __init__(out self, obj: ObjectiveFunction[Self.B], base: HestonParams):
        self.obj = obj.copy()
        self.base = base.copy()

    def __call__(self, x: List[Float64]) raises -> List[Float64]:
        return self.obj.compute(_vec_to_params(x, self.base))


struct CalibratorJacobian[B: Int](JacobianCallable):
    """Jacobian callable using finite differences."""
    var obj: ObjectiveFunction[Self.B]
    var base: HestonParams

    def __init__(out self, obj: ObjectiveFunction[Self.B], base: HestonParams):
        self.obj = obj.copy()
        self.base = base.copy()

    def __call__(self, x: List[Float64]) raises -> List[List[Float64]]:
        var r = self.obj.compute(_vec_to_params(x, self.base))
        var m = len(r)
        var n = len(x)
        var J: List[List[Float64]] = []
        for _ in range(m):
            J.append(zeros(n))

        for j in range(n):
            var eps = 1e-6 * (1.0 + abs_f64(x[j]))
            var xp = x.copy()
            var xm = x.copy()
            xp[j] = xp[j] + eps
            xm[j] = xm[j] - eps

            var rp = self.obj.compute(_vec_to_params(xp, self.base))
            var rm = self.obj.compute(_vec_to_params(xm, self.base))
            for i in range(m):
                J[i][j] = (rp[i] - rm[i]) / (2.0 * eps)
        return J^


@fieldwise_init
struct Calibrator[B: Int]:
    """Heston parameter calibration using Levenberg-Marquardt.
    B=1: single calibration. B>1: batch of B independent calibrations.
    """

    var max_iter: Int
    var tol: Float64

    def calibrate(
        self,
        market_prices: List[Float64],
        strikes: List[Float64],
        expiries: List[Float64],
        init_params: HestonParams,
    ) raises -> HestonParams:
        """Calibrate Heston params to market prices using LM optimizer."""
        var lm = LevenbergMarquardt(
            max_iter=self.max_iter,
            tol=self.tol,
            lambda_init=1e-3,
            lambda_up=10.0,
            lambda_down=0.1,
        )

        var x = _params_to_vec(init_params)

        var residual = CalibratorResidual[Self.B](
            obj=ObjectiveFunction[Self.B](market_prices.copy(), strikes.copy(), expiries.copy()),
            base=init_params,
        )
        var jacobian = CalibratorJacobian[Self.B](
            obj=ObjectiveFunction[Self.B](market_prices.copy(), strikes.copy(), expiries.copy()),
            base=init_params,
        )

        var x_opt = lm.solve(residual, jacobian, x)
        return _vec_to_params(x_opt, init_params)

    def calibrate_batch(
        self,
        market_prices_list: List[List[Float64]],
        strikes_list: List[List[Float64]],
        expiries_list: List[List[Float64]],
        init_params_list: List[HestonParams],
    ) raises -> List[HestonParams]:
        """Calibrate batch of independent parameter sets.

        Each calibration runs independently with its own market data and
        initial parameters. Uses sequential calibration per set since
        the LM optimizer internally requires sequential Jacobian evaluation.

        For GPU acceleration, each individual objective function evaluation
        within the LM optimizer uses radau_batch_solve_independent for
        batch FPE solves when computing multiple perturbation Jacobians.

        Args:
            market_prices_list: List of market price vectors (one per calibration).
            strikes_list: List of strike vectors (one per calibration).
            expiries_list: List of expiry vectors (one per calibration).
            init_params_list: List of initial Heston params (one per calibration).

        Returns:
            List of calibrated Heston params (one per calibration).
        """
        var batch_size = len(init_params_list)
        var results: List[HestonParams] = []

        for b in range(batch_size):
            var result = self.calibrate(
                market_prices_list[b],
                strikes_list[b],
                expiries_list[b],
                init_params_list[b],
            )
            results.append(result^)

        return results^
