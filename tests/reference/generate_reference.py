#!/usr/bin/env python3
"""
Generate reference data for Mojo FPE engine validation.
Run from project root: python tests/reference/generate_reference.py
Requires: numpy, scipy, cvxpy (pip install numpy scipy cvxpy)
"""

import os, sys
import numpy as np
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def save(name, **arrays):
    path = DATA_DIR / f"{name}.npz"
    np.savez(path, **arrays)
    print(f"  Saved {path.name}: {list(arrays.keys())}")


# --- Section 1: Knots ---
def gen_knots():
    print("Generating knot reference data...")
    try:
        from FPE_Solver_Final_Version import GenerateKnots

        # uniform
        n_x, d_x = 10, 3
        x_uniform = np.linspace(0.0, 1.0, n_x + d_x + 1)
        save(
            "ref_knots_uniform", knots=x_uniform, n=np.array(n_x), degree=np.array(d_x)
        )
        # non-uniform (from notebook params)
        n_x, d_x = 42, 3
        x = GenerateKnots(
            n_x,
            d_x,
            method="non-uniform",
            center=0.1,
            boundary=(50.0, 150.0),
            mean=60.0,
            std=0.1,
        ).generate_knots()
        x = x[1:]
        save("ref_knots_nonuniform", knots=x, n=np.array(n_x), degree=np.array(d_x))
    except Exception as e:
        print(f"  WARNING: knots failed: {e}")


# --- Section 2: B-spline basis ---
def gen_bspline():
    print("Generating B-spline basis reference data...")
    try:
        from FPE_Solver_Final_Version import (
            BSplineBasis,
            RecombinationBasis,
            GenerateKnots,
        )

        n_x, d_x = 10, 3
        x = np.linspace(0.0, 1.0, n_x + d_x + 1)
        basis = BSplineBasis(d_x, x)
        u = np.linspace(0.001, 0.999, 100)
        B = basis.basis_function(u)
        dB = basis.first_derivative(u)
        save(
            "ref_bspline_basis",
            basis=B.toarray(),
            knots=x,
            points=u,
            degree=np.array(d_x),
        )
        save("ref_bspline_deriv", deriv=dB.toarray(), knots=x, points=u)
        # recombination
        conds = ("newmann", "dirichlet")
        recomb = RecombinationBasis(d_x, x, conds)
        RB = recomb.basis_function(u)
        save("ref_recomb_basis", basis=RB.toarray(), knots=x, points=u)
    except Exception as e:
        print(f"  WARNING: bspline failed: {e}")


# --- Section 3: FPE matrices ---
def gen_fpe_matrices():
    print("Generating FPE matrix reference data...")
    try:
        from FPE_Solver_Final_Version import HestonSolver, GenerateKnots

        # Small grid for fast reference generation
        n_x, d_x = 10, 3
        n_v, d_v = 10, 3
        x = np.linspace(0.0, 1.0, n_x + d_x + 1)[1:]
        v = np.linspace(0.0, 1.0, n_v + d_v + 1)
        params = {
            "V_range": (0.0, 1.0),
            "S_range": (50.0, 150.0),
            "kappa": 1.2,
            "theta": 0.05,
            "sigma": 0.35,
            "V0": 0.1,
            "S0": 60.0,
        }
        degrees = [d_x, d_v]
        knots_list = [x, v]
        conditions_list = [("dirichlet", "newmann"), ("newmann", "newmann")]
        fpe = HestonSolver(degrees, knots_list, conditions_list, params=params)
        # Mass and stiffness matrices
        M = fpe.mass_matrix
        K = fpe.stiffness_matrix
        save("ref_mass_matrix", M=M.toarray(), shape=np.array(M.shape))
        save("ref_stiffness_matrix", K=K.toarray(), shape=np.array(K.shape))
    except Exception as e:
        print(f"  WARNING: FPE matrices failed: {e}")


# --- Section 4: Initial condition ---
def gen_initial_cond():
    print("Generating initial condition reference data...")
    try:
        from FPE_Solver_Final_Version import HestonSolver, GenerateKnots

        n_x, d_x = 10, 3
        n_v, d_v = 10, 3
        x = np.linspace(0.0, 1.0, n_x + d_x + 1)[1:]
        v = np.linspace(0.0, 1.0, n_v + d_v + 1)
        params = {
            "V_range": (0.0, 1.0),
            "S_range": (50.0, 150.0),
            "kappa": 1.2,
            "theta": 0.05,
            "sigma": 0.35,
            "V0": 0.1,
            "S0": 60.0,
        }
        fpe = HestonSolver(
            [d_x, d_v],
            [x, v],
            [("dirichlet", "newmann"), ("newmann", "newmann")],
            params=params,
        )
        q0 = fpe.q_initial(2.0)
        save("ref_initial_cond", q0=q0, sigma0=np.array(2.0))
    except Exception as e:
        print(f"  WARNING: initial condition failed: {e}")


# --- Section 5: ODE solution ---
def gen_ode_solution():
    print("Generating ODE solution reference data...")
    try:
        from FPE_Solver_Final_Version import HestonSolver

        n_x, d_x = 10, 3
        n_v, d_v = 10, 3
        x = np.linspace(0.0, 1.0, n_x + d_x + 1)[1:]
        v = np.linspace(0.0, 1.0, n_v + d_v + 1)
        params = {
            "V_range": (0.0, 1.0),
            "S_range": (50.0, 150.0),
            "kappa": 1.2,
            "theta": 0.05,
            "sigma": 0.35,
            "V0": 0.1,
            "S0": 60.0,
        }
        fpe = HestonSolver(
            [d_x, d_v],
            [x, v],
            [("dirichlet", "newmann"), ("newmann", "newmann")],
            params=params,
        )
        delta = fpe.delta_approx(2.0)
        save("ref_ode_solution", q_t=delta, t=np.array([0.0, 0.5, 1.0]))
    except Exception as e:
        print(f"  WARNING: ODE solution failed: {e}")


# --- Section 6: PDF grid p(S, V, T) ---
def gen_pdf_grid():
    print("Generating PDF grid reference data...")
    try:
        from FPE_Solver_Final_Version import HestonSolver, GenerateKnots

        n_x, d_x = 10, 3
        n_v, d_v = 10, 3
        x = np.linspace(0.0, 1.0, n_x + d_x + 1)[1:]
        v = np.linspace(0.0, 1.0, n_v + d_v + 1)
        params = {
            "V_range": (0.0, 1.0),
            "S_range": (50.0, 150.0),
            "kappa": 1.2,
            "theta": 0.05,
            "sigma": 0.35,
            "V0": 0.1,
            "S0": 60.0,
            "T": 0.5,
        }
        fpe = HestonSolver(
            [d_x, d_v],
            [x, v],
            [("dirichlet", "newmann"), ("newmann", "newmann")],
            params=params,
        )
        time_eval = np.array([0.0, 0.25, 0.5])
        pdf_2d, t = fpe.fpe_solver(sigma0=2.0, time=time_eval)
        save(
            "ref_pdf_grid",
            pdf=pdf_2d,
            t=t,
            s_points=fpe.s_points,
            v_points=fpe.v_points,
            sigma0=np.array(2.0),
        )
    except Exception as e:
        print(f"  WARNING: PDF grid failed: {e}")


# --- Section 7: Barrier option price + delta + gamma ---
def gen_barrier_price():
    print("Generating barrier option price reference data...")
    try:
        from FPE_Solver_Final_Version import HestonSolver, GenerateKnots

        n_x, d_x = 10, 3
        n_v, d_v = 10, 3

        asset_range = (50.0, 150.0)
        var_range = (0.0, 1.0)
        S0_center = 60.0
        sigma0 = 2.0
        rate = 0.1
        maturity = 0.5
        time_eval = np.array([0.0, maturity])
        strikes = np.linspace(55.0, 75.0, 5)

        # Build a small solver for three S0 offsets to compute delta/gamma via FD
        results = {}
        for s0_val, label in [(59.5, "lo"), (60.0, "mid"), (60.5, "hi")]:
            x_knots = np.linspace(0.0, 1.0, n_x + d_x + 1)[1:]
            v_knots = np.linspace(0.0, 1.0, n_v + d_v + 1)
            params = {
                "V_range": var_range,
                "S_range": asset_range,
                "kappa": 1.2,
                "theta": 0.05,
                "sigma": 0.35,
                "V0": 0.1,
                "S0": s0_val,
                "T": maturity,
            }
            fpe = HestonSolver(
                [d_x, d_v],
                [x_knots, v_knots],
                [("dirichlet", "newmann"), ("newmann", "newmann")],
                params=params,
            )
            # Solve FPE and get marginal over v at maturity
            pdf_2d, t_out = fpe.fpe_solver(sigma0=sigma0, time=time_eval)
            # pdf_2d shape: (n_s, n_v, n_t) — take terminal slice
            pdf_T = pdf_2d[:, :, -1]
            # Integrate over v to get marginal density over S
            marginal_s = pdf_T @ fpe.nodes_weights["v_weights"]
            results[label] = {
                "s_points": fpe.s_points,
                "v_weights": fpe.nodes_weights["v_weights"],
                "marginal_s": marginal_s,
            }

        def call_price_vec(marginal_s, s_points, strikes):
            """Compute undiscounted call prices for an array of strikes."""
            prices = np.zeros(len(strikes))
            ds = np.diff(s_points)
            # Use trapezoidal weights for integration over S
            trap_w = np.concatenate(([ds[0] / 2], (ds[:-1] + ds[1:]) / 2, [ds[-1] / 2]))
            for j, K in enumerate(strikes):
                payoff = np.maximum(s_points - K, 0.0)
                prices[j] = np.dot(payoff * marginal_s, trap_w)
            return prices

        s_mid = results["mid"]["s_points"]
        price_lo = call_price_vec(
            results["lo"]["marginal_s"], results["lo"]["s_points"], strikes
        )
        price_mid = call_price_vec(results["mid"]["marginal_s"], s_mid, strikes)
        price_hi = call_price_vec(
            results["hi"]["marginal_s"], results["hi"]["s_points"], strikes
        )

        delta = (price_hi - price_lo) / 1.0  # dV/dS  (dS = 0.5 * 2 = 1.0)
        gamma = (price_hi - 2.0 * price_mid + price_lo) / (0.5**2)  # d²V/dS²

        save(
            "ref_barrier_price",
            price=price_mid,
            delta=delta,
            gamma=gamma,
            strikes=strikes,
            S0=np.array(S0_center),
            sigma0=np.array(sigma0),
            rate=np.array(rate),
            maturity=np.array(maturity),
        )
    except Exception as e:
        print(f"  WARNING: barrier price failed: {e}")


# --- Section 8: NAIS Volterra + variance ---
def gen_nais_processes():
    print("Generating NAIS process reference data...")
    try:
        import sys

        sys.path.insert(0, str(ROOT))
        # Minimal FBSNN subclass to access volterra/variance
        import numpy as np
        from NAIS_rBM import FBSNN
        import tensorflow as tf

        class MinimalFBSNN(FBSNN):
            def F_tf(self, t, X, Y, Z, Tilde_Z):
                return tf.zeros_like(Y)

            def g_tf(self, X):
                return tf.zeros([self.M, 1])

            def mu_tf(self, t, X, Y, Du):
                return tf.zeros([self.M, self.D])

        Xi = np.array([[0.0]])
        model = MinimalFBSNN(
            Xi=Xi,
            T=1.0,
            M=4,
            N=10,
            D=1,
            H=0.7,
            eta=0.3,
            pho=-0.4,
            r=0.05,
            layers=[2, 16, 16, 16, 16, 1, 1],
        )
        t, W, tilde_X = model.volterra()
        W_s, var = model.variance()
        save("ref_nais_volterra", t=t, W=W, tilde_X=tilde_X)
        save("ref_nais_variance", W_s=W_s, variance=var)
    except Exception as e:
        print(f"  WARNING: NAIS processes failed: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("FPE Engine — Reference Data Generator")
    print("=" * 60)
    gen_knots()
    gen_bspline()
    gen_fpe_matrices()
    gen_initial_cond()
    gen_ode_solution()
    gen_pdf_grid()
    gen_barrier_price()
    gen_nais_processes()
    print("=" * 60)
    print("Reference data generation complete.")
    print(f"Files saved to: {DATA_DIR}")
