trait Payoff:
    def __init__(out self):
        ...

    def evaluate(self, S: Float64, K: Float64, barrier: Float64) -> Float64:
        ...

    def name(self) -> StaticString:
        ...


struct BarrierUpAndOut(Payoff):
    """Up-and-out call: max(S-K, 0) * I(S < barrier)."""

    def __init__(out self):
        pass

    def evaluate(self, S: Float64, K: Float64, barrier: Float64) -> Float64:
        if S >= barrier:
            return 0.0
        var intrinsic = S - K
        if intrinsic > 0.0:
            return intrinsic
        return 0.0

    def name(self) -> StaticString:
        return "BarrierUpAndOut"


struct BarrierDownAndIn(Payoff):
    """Down-and-in put: max(K-S, 0) * I(S <= barrier)."""

    def __init__(out self):
        pass

    def evaluate(self, S: Float64, K: Float64, barrier: Float64) -> Float64:
        if S > barrier:
            return 0.0
        var intrinsic = K - S
        if intrinsic > 0.0:
            return intrinsic
        return 0.0

    def name(self) -> StaticString:
        return "BarrierDownAndIn"


struct EuropeanCall(Payoff):
    """European call: max(S-K, 0)."""

    def __init__(out self):
        pass

    def evaluate(self, S: Float64, K: Float64, barrier: Float64) -> Float64:
        _ = barrier
        var intrinsic = S - K
        if intrinsic > 0.0:
            return intrinsic
        return 0.0

    def name(self) -> StaticString:
        return "EuropeanCall"


struct EuropeanPut(Payoff):
    """European put: max(K-S, 0)."""

    def __init__(out self):
        pass

    def evaluate(self, S: Float64, K: Float64, barrier: Float64) -> Float64:
        _ = barrier
        var intrinsic = K - S
        if intrinsic > 0.0:
            return intrinsic
        return 0.0

    def name(self) -> StaticString:
        return "EuropeanPut"
