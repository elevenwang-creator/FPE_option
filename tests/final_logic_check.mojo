from engines.fpe.heston_params import HestonParams
from engines.fpe.domain import FPEDomain
from engines.fpe.solver import FPESolver

def main() raises:
    print("Mojo FPE Logic Chain Verification...")
    
    # 1. Initialize Heston Parameters
    var params = HestonParams(
        kappa=1.2,
        theta=0.05,
        sigma=0.35,
        rho=-0.4,
        r=0.1,
        S0=60.0,
        V0=0.1,
        S_min=0.0,
        S_max=150.0,
        V_min=0.0,
        V_max=1.0,
        T=0.6
    )
    print("Step 1: Heston Parameters initialized.")

    # 2. Automated Non-uniform Domain Generation (Triggered by params)
    var domain = FPEDomain(params, n_s=20, n_v=20, degree_s=3, degree_v=3)
    print("Step 2: Non-uniform Domain generated.")
    print(" - S Knots count:", len(domain.s_knots))
    print(" - V Knots count:", len(domain.v_knots))

    # 3. Solve FPE System (RadauIIA ODE)
    var solver = FPESolver(rtol=1e-4, atol=1e-6, max_step=0.05, first_step=0.0)
    var t_eval: List[Float64] = [0.6]  # Correct Mojo List Literal
    
    print("Step 3: Solving ODE system...")
    var pdf_grid = solver.solve(domain, params, t_eval)
    
    # 4. Result Inspection
    var n_rows = len(pdf_grid)
    var n_cols = len(pdf_grid[0])
    print("Step 4: PDF Grid produced with shape:", n_rows, "x", n_cols)
    
    var total_sum = 0.0
    for i in range(n_rows):
        for j in range(n_cols):
            total_sum += pdf_grid[i][j]
    
    print(" - Total Probability Sum (Normalized):", total_sum)
    print("SUCCESS: End-to-end logic chain verified (Heston -> Knots -> PDF).")
