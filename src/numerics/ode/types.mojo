trait ODESystem:
    """Interface for ODE right-hand side: dy/dt = f(t, y)."""

    def rhs(self, t: Float64, y: List[Float64], mut dydt: List[Float64]) raises:
        ...

    def dim(self) -> Int:
        ...


@fieldwise_init
struct ODESolution(Copyable, Movable):
    var t: List[Float64]
    var y: List[List[Float64]]
    var success: Bool
    var message: String
