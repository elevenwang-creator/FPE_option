@fieldwise_init
struct HestonParams(Copyable, Movable, Writable, Hashable):
    var kappa: Float64
    var theta: Float64
    var sigma: Float64
    var rho: Float64
    var r: Float64
    var T: Float64
    var S0: Float64
    var V0: Float64
    var S_min: Float64
    var S_max: Float64
    var V_min: Float64
    var V_max: Float64

    def feller_condition(self) -> Float64:
        return 2.0 * self.kappa * self.theta / (self.sigma * self.sigma) - 1.0

    def is_valid(self) -> Bool:
        return (
            self.kappa > 0.0
            and self.theta > 0.0
            and self.sigma > 0.0
            and self.T > 0.0
            and self.S_max > self.S_min
            and self.V_max > self.V_min
        )


struct HestonParamsBatch[B: Int](Copyable, Movable):
    """Packed Structural Array (SoA) layout for Heston Parameters using InlineArray limiters."""
    var kappa: InlineArray[Float64, Self.B]
    var theta: InlineArray[Float64, Self.B]
    var sigma: InlineArray[Float64, Self.B]
    var rho: InlineArray[Float64, Self.B]
    var r: InlineArray[Float64, Self.B]
    var T: InlineArray[Float64, Self.B]
    var S0: InlineArray[Float64, Self.B]
    var V0: InlineArray[Float64, Self.B]
    var S_min: InlineArray[Float64, Self.B]
    var S_max: InlineArray[Float64, Self.B]
    var V_min: InlineArray[Float64, Self.B]
    var V_max: InlineArray[Float64, Self.B]

    def __init__(out self, params: List[HestonParams]):
        self.kappa = InlineArray[Float64, Self.B](fill=0.0)
        self.theta = InlineArray[Float64, Self.B](fill=0.0)
        self.sigma = InlineArray[Float64, Self.B](fill=0.0)
        self.rho = InlineArray[Float64, Self.B](fill=0.0)
        self.r = InlineArray[Float64, Self.B](fill=0.0)
        self.T = InlineArray[Float64, Self.B](fill=0.0)
        self.S0 = InlineArray[Float64, Self.B](fill=0.0)
        self.V0 = InlineArray[Float64, Self.B](fill=0.0)
        self.S_min = InlineArray[Float64, Self.B](fill=0.0)
        self.S_max = InlineArray[Float64, Self.B](fill=0.0)
        self.V_min = InlineArray[Float64, Self.B](fill=0.0)
        self.V_max = InlineArray[Float64, Self.B](fill=0.0)

        for i in range(len(params)):
            if i >= Self.B: break
            self.kappa[i] = params[i].kappa
            self.theta[i] = params[i].theta
            self.sigma[i] = params[i].sigma
            self.rho[i] = params[i].rho
            self.r[i] = params[i].r
            self.T[i] = params[i].T
            self.S0[i] = params[i].S0
            self.V0[i] = params[i].V0
            self.S_min[i] = params[i].S_min
            self.S_max[i] = params[i].S_max
            self.V_min[i] = params[i].V_min
            self.V_max[i] = params[i].V_max
