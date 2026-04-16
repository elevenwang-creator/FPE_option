from numerics.ode.radau import RadauIIA
from numerics.ode.types import ODESystem, ODESolution
from numerics.utils import abs_f64
from std.math import exp


struct TestSystem(ODESystem):
    var alpha: Float64
    
    def __init__(out self, alpha: Float64):
        self.alpha = alpha
    
    def dim(self) -> Int:
        return 2
    
    def rhs(self, t: Float64, y: List[Float64], out dy: List[Float64]) raises:
        dy[0] = self.alpha * y[0] - y[1]
        dy[1] = y[0] - self.alpha * y[1]


struct SimpleDecay(ODESystem):
    var lam: Float64
    
    def __init__(out self, lam: Float64):
        self.lam = lam
    
    def dim(self) -> Int:
        return 1
    
    def rhs(self, t: Float64, y: List[Float64], out dy: List[Float64]) raises:
        dy[0] = -self.lam * y[0]


def main() raises:
    print("Test 1: Simple exponential decay y' = -lambda*y")
    print("=" * 50)
    var sys = SimpleDecay(lam=1.0)
    var solver = RadauIIA[SimpleDecay](
        rtol=1e-8, atol=1e-10, max_step=0.1,
        newton_tol=1e-8, newton_max_iter=12
    )
    var y0: List[Float64] = [1.0]
    var sol = solver.solve(sys, (0.0, 2.0), y0^)
    
    print("Success: " + String(sol.success))
    print("Message: " + sol.message)
    print("Time points: " + String(len(sol.t)))
    if len(sol.t) > 0:
        print("Initial y: " + String(sol.y[0][0]))
        var last_t_idx = len(sol.t) - 1
        print("Final y: " + String(sol.y[0][last_t_idx]))
        print("Expected final y = exp(-2.0) = " + String(exp(-2.0)))
        var err = abs_f64(sol.y[0][last_t_idx] - exp(-2.0))
        print("Error: " + String(err))
    print("")
    
    print("Test 2: 2D rotation system")
    print("=" * 50)
    var sys2 = TestSystem(alpha=0.0)
    var solver2 = RadauIIA[TestSystem](
        rtol=1e-8, atol=1e-10, max_step=0.1,
        newton_tol=1e-8, newton_max_iter=12
    )
    var y0_2: List[Float64] = [1.0, 0.0]
    var sol2 = solver2.solve(sys2, (0.0, 3.14159), y0_2^)
    
    print("Success: " + String(sol2.success))
    print("Message: " + sol2.message)
    print("Time points: " + String(len(sol2.t)))
    if len(sol2.t) > 0:
        print("Initial: (" + String(sol2.y[0]) + ", " + String(sol2.y[1]) + ")")
        var last_idx = len(sol2.y) - 1
        print("Final: (" + String(sol2.y[last_idx - 1]) + ", " + String(sol2.y[last_idx]) + ")")
        print("Expected: (~-1, ~0) after pi rotation")
    print("")
    
    print("Test complete.")
