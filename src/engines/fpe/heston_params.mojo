from std.math import exp, sqrt


@fieldwise_init
struct HestonParams(Copyable, Hashable, Movable, Writable):
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
            and self.S0 > 0.0
            and self.V0 >= 0.0
            and self.S_max > self.S_min
            and self.V_max > self.V_min
            and self.rho >= -1.0
            and self.rho <= 1.0
        )

    def validate(self) raises:
        """Validate Heston parameters and raise Error if invalid."""
        if self.kappa <= 0.0:
            raise Error("kappa must be positive, got " + String(self.kappa))
        if self.theta <= 0.0:
            raise Error("theta must be positive, got " + String(self.theta))
        if self.sigma <= 0.0:
            raise Error("sigma must be positive, got " + String(self.sigma))
        if self.T <= 0.0:
            raise Error("T must be positive, got " + String(self.T))
        if self.S0 <= 0.0:
            raise Error("S0 must be positive, got " + String(self.S0))
        if self.V0 < 0.0:
            raise Error("V0 must be non-negative, got " + String(self.V0))
        if self.rho < -1.0 or self.rho > 1.0:
            raise Error("rho must be in [-1, 1], got " + String(self.rho))
        if self.S_max <= self.S_min:
            raise Error("S_max must be > S_min")
        if self.V_max <= self.V_min:
            raise Error("V_max must be > V_min")

    def recommended_S_min(self) -> Float64:
        var vol_scale = sqrt(self.V0) * sqrt(self.T)
        return self.S0 * exp(-6.0 * vol_scale)

    def recommended_S_max(self) -> Float64:
        var vol_scale = sqrt(self.V0) * sqrt(self.T)
        return self.S0 * exp(6.0 * vol_scale)

