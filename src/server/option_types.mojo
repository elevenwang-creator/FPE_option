from engines.fpe.heston_params import HestonParams


@fieldwise_init
struct FpeParams(Copyable, Movable, Writable):
    var heston: HestonParams
    var n_s: Int
    var n_v: Int
    var barrier: Float64
    var option_type: Int
    var strikes: List[Float64]

    def is_valid(self) -> Bool:
        if not self.heston.is_valid():
            return False
        if self.option_type < 0 or self.option_type > 9:
            return False
        if len(self.strikes) == 0:
            return False
        for k in self.strikes:
            if k <= 0.0:
                return False
        if self.option_type <= 3:
            if self.barrier <= 0.0 or self.barrier >= self.heston.S0:
                return False
        elif self.option_type <= 7:
            if self.barrier <= 0.0 or self.barrier <= self.heston.S0:
                return False
        else:
            if self.barrier != 0.0:
                return False
        return True

    def revised_heston(self) -> HestonParams:
        var s_min = self.heston.S_min
        var s_max = self.heston.S_max
        if self.option_type <= 3:
            s_min = self.barrier
        elif self.option_type <= 7:
            s_max = self.barrier
        return HestonParams(
            kappa=self.heston.kappa,
            theta=self.heston.theta,
            sigma=self.heston.sigma,
            rho=self.heston.rho,
            r=self.heston.r,
            T=self.heston.T,
            S0=self.heston.S0,
            V0=self.heston.V0,
            S_min=s_min,
            S_max=s_max,
            V_min=0.0,
            V_max=1.0,
        )

    def s_left_cond(self) -> String:
        if self.option_type <= 3:
            return "dirichlet"
        elif self.option_type <= 7:
            return "neumann"
        else:
            return "dirichlet"

    def s_right_cond(self) -> String:
        if self.option_type <= 3:
            return "neumann"
        elif self.option_type <= 7:
            return "dirichlet"
        else:
            return "neumann"


@fieldwise_init
struct PricingResult(Copyable, Movable, Writable):
    var price: Float64
    var delta: Float64
    var gamma: Float64
    var vega: Float64
    var success: Bool
