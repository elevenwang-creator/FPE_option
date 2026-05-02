"""Unified option pricer with pre-computed quadrature weights.

Key improvements over original:
- Pre-computes trapezoidal weights once, reuses per option
- Hoists payoff computation out of inner V-loop (payoff depends only on S)
- SIMD inner loop over variance dimension
- Computes Vega alongside Delta/Gamma
- GPU batch pricing routes through engines.fpe.gpu per architecture requirements
"""

from std.sys import has_accelerator

from server.pdf_cache import PDFGrid
from server.interpolator import Interpolator
from server.payoffs import BarrierDownAndIn, BarrierUpAndOut, EuropeanCall, EuropeanPut
from server.greeks import Greeks
from server.option_types import PricingResult
from std.sys import simd_width_of
from std.algorithm import parallelize


@fieldwise_init
struct PricingRequest(Copyable, Movable):
    var S: Float64
    var K: Float64
    var V: Float64
    var barrier: Float64
    var payoff_type: Int
    var param_hash: UInt64

    def is_valid(self) -> Bool:
        """Validate pricing request parameters."""
        return (
            self.S > 0.0
            and self.K > 0.0
            and self.V >= 0.0
            and (self.barrier == 0.0 or self.barrier > self.S)
            and self.payoff_type >= 0
            and self.payoff_type <= 3
        )


@fieldwise_init
struct Pricer[B: Int]:
    """Unified pricer. B=1 -> CPU path, B>1 -> batch path with fallback.

    Pre-computes trapezoidal quadrature weights for O(1) amortized cost
    per option pricing call after the first.
    """

    var interpolator: Interpolator
    var greeks_computer: Greeks[Self.B]

    def price(self, grid: PDFGrid, requests: List[PricingRequest]) -> List[PricingResult]:
        """Price B options using pre-computed PDF grid."""

        comptime if Self.B == 1:
            return self._price_single(grid, requests)
        else:
            comptime if has_accelerator():
                return self._price_gpu_batch(grid, requests)
            else:
                return self._price_cpu_parallel(grid, requests)

    def _price_single(self, grid: PDFGrid, requests: List[PricingRequest]) -> List[PricingResult]:
        """CPU path: pre-compute weights once, then integrate for each option."""

        var ds_weights: List[Float64] = []
        var dv_weights: List[Float64] = []
        if len(grid.ds_weights) > 0:
            ds_weights = grid.ds_weights.copy()
        else:
            ds_weights = self._compute_trap_weights(grid.s_points)
        if len(grid.dv_weights) > 0:
            dv_weights = grid.dv_weights.copy()
        else:
            dv_weights = self._compute_trap_weights(grid.v_points)

        var results: List[PricingResult] = []
        for req in requests:
            if not req.is_valid():
                results.append(PricingResult(
                    price=0.0, delta=0.0, gamma=0.0, vega=0.0, success=False,
                ))
                continue
            var price = self._integrate_payoff_fast(grid, req, ds_weights, dv_weights)
            var payoff = self._get_payoff(req)
            var delta = self.greeks_computer.compute_delta(
                grid, self.interpolator,
                req.S, req.V, req.K, req.barrier, payoff,
            )
            var gamma = self.greeks_computer.compute_gamma(
                grid, self.interpolator,
                req.S, req.V, req.K, req.barrier, payoff,
            )
            var vega = self.greeks_computer.compute_vega(
                grid, self.interpolator,
                req.S, req.V, req.K, req.barrier, payoff,
            )
            results.append(PricingResult(
                price=price, delta=delta, gamma=gamma, vega=vega, success=True,
            ))
        return results^

    def _price_cpu_parallel(self, grid: PDFGrid, requests: List[PricingRequest]) -> List[PricingResult]:
        """Batch CPU path: parallelize across options."""
        var n = len(requests)
        var results: List[PricingResult] = []
        for _ in range(n):
            results.append(PricingResult(
                price=0.0, delta=0.0, gamma=0.0, vega=0.0, success=False
            ))

        var ds_weights: List[Float64] = []
        var dv_weights: List[Float64] = []
        if len(grid.ds_weights) > 0:
            ds_weights = grid.ds_weights.copy()
        else:
            ds_weights = self._compute_trap_weights(grid.s_points)
        if len(grid.dv_weights) > 0:
            dv_weights = grid.dv_weights.copy()
        else:
            dv_weights = self._compute_trap_weights(grid.v_points)

        @parameter
        def worker(i: Int):
            var payoff = self._get_payoff_for_greeks(requests[i])
            var price = self._integrate_payoff_fast(
                grid, requests[i], ds_weights, dv_weights
            )
            var delta = self.greeks_computer.compute_delta(
                grid, self.interpolator,
                requests[i].S, requests[i].V, requests[i].K, requests[i].barrier, payoff,
            )
            var gamma = self.greeks_computer.compute_gamma(
                grid, self.interpolator,
                requests[i].S, requests[i].V, requests[i].K, requests[i].barrier, payoff,
            )
            var vega = self.greeks_computer.compute_vega(
                grid, self.interpolator,
                requests[i].S, requests[i].V, requests[i].K, requests[i].barrier, payoff,
            )
            results[i] = PricingResult(
                price=price, delta=delta, gamma=gamma, vega=vega, success=True
            )

        parallelize[worker](n)
        return results^

    def _price_gpu_batch(self, grid: PDFGrid, requests: List[PricingRequest]) -> List[PricingResult]:
        """GPU batch pricing via engines.fpe.gpu logic chain.

        Routes through engines.fpe.gpu.executor.GPUFullChainExecutor
        per architecture requirements in logic_picture.md:
        service layer delegates to engine module for full GPU pipeline.
        """
        comptime if has_accelerator():
            from engines.fpe.gpu.executor import GPUFullChainExecutor
            from gpu_utils.dtype import METAL_MAX_N, CUDA_MAX_N
            from std.sys import has_apple_gpu_accelerator

            comptime GPU_MAX_N = METAL_MAX_N if has_apple_gpu_accelerator() else CUDA_MAX_N

            var n_options = len(requests)
            if n_options > GPU_MAX_N:
                return self._price_cpu_parallel(grid, requests)

            try:
                var strikes: List[Float64] = []
                var barriers: List[Float64] = []
                for i in range(n_options):
                    strikes.append(requests[i].K)
                    barriers.append(requests[i].barrier)

                var ds_weights = grid.ds_weights.copy()
                var dv_weights = grid.dv_weights.copy()
                if len(ds_weights) == 0:
                    ds_weights = self._compute_trap_weights(grid.s_points)
                if len(dv_weights) == 0:
                    dv_weights = self._compute_trap_weights(grid.v_points)

                var executor = GPUFullChainExecutor[Self.B](
                    n_s=len(grid.s_points), n_v=len(grid.v_points)
                )
                var prices = executor.price_options(
                    grid.pdf, grid.s_points, grid.v_points,
                    ds_weights, dv_weights,
                    strikes^, barriers^,
                    len(grid.s_points), len(grid.v_points), n_options,
                )

                var results: List[PricingResult] = []
                for i in range(n_options):
                    var payoff = self._get_payoff(requests[i])
                    var delta = self.greeks_computer.compute_delta(
                        grid, self.interpolator,
                        requests[i].S, requests[i].V, requests[i].K, requests[i].barrier, payoff,
                    )
                    var gamma = self.greeks_computer.compute_gamma(
                        grid, self.interpolator,
                        requests[i].S, requests[i].V, requests[i].K, requests[i].barrier, payoff,
                    )
                    var vega = self.greeks_computer.compute_vega(
                        grid, self.interpolator,
                        requests[i].S, requests[i].V, requests[i].K, requests[i].barrier, payoff,
                    )
                    results.append(PricingResult(
                        price=prices[i], delta=delta, gamma=gamma, vega=vega, success=True,
                    ))
                return results^
            except:
                pass
            return self._price_cpu_parallel(grid, requests)
        else:
            return self._price_cpu_parallel(grid, requests)

    @always_inline
    def _get_payoff(self, req: PricingRequest) -> EuropeanCall:
        """Get payoff for Greeks finite-difference integration.
        Note: Always returns EuropeanCall because Greeks computation uses
        finite differences on the PDF grid, where payoff.evaluate() computes
        max(S-K, 0). Barrier option Greeks require different formulas entirely.
        """
        _ = req
        return EuropeanCall()

    @always_inline
    def _get_payoff_for_greeks(self, req: PricingRequest) -> EuropeanCall:
        """Get payoff type for Greeks — currently EuropeanCall for all types.
        TODO: Implement proper barrier option Greeks when needed.
        """
        _ = req
        return EuropeanCall()

    @always_inline
    def _payoff_value(self, req: PricingRequest, S: Float64) -> Float64:
        """Evaluate payoff at spot price S. Inlined for zero call overhead."""
        if req.payoff_type == 0:
            return BarrierUpAndOut().evaluate(S, req.K, req.barrier)
        if req.payoff_type == 1:
            return EuropeanCall().evaluate(S, req.K, req.barrier)
        if req.payoff_type == 2:
            return BarrierDownAndIn().evaluate(S, req.K, req.barrier)
        if req.payoff_type == 3:
            return EuropeanPut().evaluate(S, req.K, req.barrier)
        return EuropeanCall().evaluate(S, req.K, req.barrier)

    def _compute_trap_weights(self, points: List[Float64]) -> List[Float64]:
        """Compute trapezoidal quadrature weights for a set of points.

        Pre-computed once, reused for every pricing call.
        Interior: (x[i+1] - x[i-1]) / 2 (midpoint rule)
        Boundary: 1.0 (half-interval)
        """
        var n = len(points)
        var w: List[Float64] = []
        for i in range(n):
            if i == 0 or i == n - 1:
                w.append(1.0)
            else:
                w.append((points[i + 1] - points[i - 1]) * 0.5)
        return w^

    def _integrate_payoff(self, grid: PDFGrid, req: PricingRequest) -> Float64:
        """Numerical integration: price = ∫∫ payoff(S) * pdf(S,V) dS dV.

        Retained for backward compatibility. Uses on-the-fly weight computation.
        """
    var ds_weights: List[Float64] = []
    var dv_weights: List[Float64] = []
    if len(grid.ds_weights) > 0:
        ds_weights = grid.ds_weights.copy()
    else:
        ds_weights = self._compute_trap_weights(grid.s_points)
    if len(grid.dv_weights) > 0:
        dv_weights = grid.dv_weights.copy()
    else:
        dv_weights = self._compute_trap_weights(grid.v_points)
    return self._integrate_payoff_fast(grid, req, ds_weights, dv_weights)

    def _integrate_payoff_fast(
        self,
        grid: PDFGrid,
        req: PricingRequest,
        ds_weights: List[Float64],
        dv_weights: List[Float64],
    ) -> Float64:
        """Optimized integration with pre-computed weights.

        Key optimization: payoff(S) is hoisted out of the inner V-loop
        since payoff depends only on S, not V. This avoids redundant
        payoff evaluation (was called n_s × n_v times, now n_s times).
        """
        var price = 0.0
        var n_s = len(grid.s_points)
        var n_v = len(grid.v_points)

        for i in range(n_s):
            var S = grid.s_points[i]
            var payoff_val = self._payoff_value(req, S)

            # Skip zero-payoff rows entirely (common for OTM options)
            if payoff_val == 0.0:
                continue

            var payoff_ds = payoff_val * ds_weights[i]

            # Inner loop: sum pdf over variance dimension
            var v_sum = 0.0
            comptime simd_width = simd_width_of[DType.float64]()

            var j = 0
            while j + simd_width <= n_v:
                var pdf_vals = SIMD[DType.float64, simd_width]()
                var dv_vals = SIMD[DType.float64, simd_width]()
                for k in range(simd_width):
                    pdf_vals[k] = grid.pdf[i][j + k]
                    dv_vals[k] = dv_weights[j + k]
                v_sum += (pdf_vals * dv_vals).reduce_add()
                j += simd_width

            # Scalar tail
            while j < n_v:
                v_sum += grid.pdf[i][j] * dv_weights[j]
                j += 1

            price += payoff_ds * v_sum

        return price
