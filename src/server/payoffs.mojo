trait Payoff:
    def __init__(out self, option_type: Int, var strikes: List[Float64], barrier: Float64): ...

    def evaluate(self, S: Float64) -> List[Float64]: ...

    def name(self) -> StaticString: ...


struct BarrierPayoff(Payoff):
    var option_type: Int
    var strikes: List[Float64]
    var barrier: Float64

    def __init__(out self, option_type: Int, var strikes: List[Float64], barrier: Float64):
        self.option_type = option_type
        self.strikes = strikes^
        self.barrier = barrier

    def evaluate(self, S: Float64) -> List[Float64]:
        var is_call = (self.option_type % 2 == 0)
        var active = self._is_active(S)
        var result: List[Float64] = []
        for i in range(len(self.strikes)):
            if not active:
                result.append(0.0)
            elif is_call:
                var val = S - self.strikes[i]
                if val > 0.0:
                    result.append(val)
                else:
                    result.append(0.0)
            else:
                var val = self.strikes[i] - S
                if val > 0.0:
                    result.append(val)
                else:
                    result.append(0.0)
        return result^

    def _is_active(self, S: Float64) -> Bool:
        if self.option_type == 0 or self.option_type == 1:
            return S <= self.barrier
        elif self.option_type == 2 or self.option_type == 3:
            return S > self.barrier
        elif self.option_type == 4 or self.option_type == 5:
            return S >= self.barrier
        elif self.option_type == 6 or self.option_type == 7:
            return S < self.barrier
        else:
            return True

    def name(self) -> StaticString:
        return "BarrierPayoff"
