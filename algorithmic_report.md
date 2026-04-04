# Algorithmic Report: FPE & NAIS-Net Option Pricing Engine

This report details the mathematical and algorithmic foundations of the Fokker-Planck Equation (FPE) solver and the Neural Adaptive Iterative Solver (NAIS-Net) as implemented in this codebase.

## 1. Fokker-Planck Equation (FPE) Solver

The primary engine for Heston model pricing is a numerical solver for the FPE, which describes the evolution of the joint probability density function (PDF) of the asset price and its variance.

### A. Mathematical Model
*   **Process**: Heston Stochastic Volatility Model.
*   **Equation**: The Kolmogorov Forward Equation (FPE) for the PDF $p(s, v, t)$:
    $$\frac{\partial p}{\partial t} = -\nabla \cdot \mathbf{J}$$
    where $\mathbf{J}$ is the probability current containing drift and diffusion terms.

### B. Galerkin Variational Method
Instead of standard Finite Difference Methods, the solver uses a **Galerkin approach with B-spline basis functions**:
1.  **Approximation**: $p(s, v, t) \approx \sum_{i=1}^{n} \varphi_i(s, v) q_i(t)$.
2.  **Basis**: 2D B-splines $\varphi_i(s, v)$ formed by the tensor product of 1D B-splines.
3.  **Knot Strategy**: `GenerateKnots` implements a non-uniform distribution using parabolic functions and Chebyshev nodes to cluster points near $(S_0, V_0)$ and the absorption/reflection boundaries.

### C. Matrix System
The variational formulation transforms the PDE into a system of Ordinary Differential Equations (ODEs):
$$\mathbf{M}\dot{\mathbf{q}}(t) + \mathbf{K}\mathbf{q}(t) = \mathbf{0}$$
*   **Mass Matrix ($\mathbf{M}$)**: $\int_{\Omega} \varphi^T \varphi \, d\Omega$.
*   **Stiffness Matrix ($\mathbf{K}$)**: Derived from the drift and diffusion operators. The implementation in `HestonSolver` calculates this using 8 specific sub-matrices ($K_1$ through $K_8$) representing individual terms of the Heston operator.

### D. Initial Condition & Constraints
To handle the Dirac delta initial condition:
*   **Optimization**: Uses the **OSQP** solver to find initial coefficients $\mathbf{q}(0)$ that minimize the squared error relative to a narrow Gaussian approximation, subject to:
    *   **Positivity**: $\mathbf{q} \geq 0$.
    *   **Unit Integral**: $\int p \, d\Omega = 1$.

### E. Temporal Integration
*   **RadauIIA**: A 5th-order implicit Runge-Kutta method is used for time-stepping, providing high stability for the stiff systems typical of Galerkin discretizations.

---

## 2. NAIS-Net (Neural FBSDE Solver)

For high-dimensional pricing or rough volatility models (like Rough Bergomi), the engine employs a Deep Learning approach based on Forward-Backward SDEs (FBSDE).

### A. Architecture (Stable NAIS-Net)
*   **Stability**: Uses `StableLinear` layers that enforce a weight constraint to ensure spectral stability of the mapping. This prevents exploding gradients during the long-sequence integration of the SDE.
*   **Residual Flow**: The network architecture mimics the flow of a discretization scheme, allowing it to learn the value function (price) and its gradient (delta/Greeks) simultaneously.

### B. Rough Volatility Simulation
*   **Volterra Process**: Simulates fractional Brownian Motion (fBM) using a **Hybrid Scheme**.
*   **FFT Optimization**: Employs FFT-based convolution to compute the weighted noise kernel efficiently.

---

## 3. Mojo Rewrite Strategy (Performance Targets)

The implementation plan highlights the transition from the current Python/TensorFlow stack to native Mojo/MAX:
1.  **Recursion**: De Boor-Cox algorithm will be vectorized via SIMD.
2.  **Assembly**: Sparse matrix construction will move from serial Python loops to parallelized Mojo kernels.
3.  **Matmul**: Neural network layers will use **MAX AI Kernels** for production-grade throughput.
4.  **Target**: Sub-millisecond single pricing for live trading and high-throughput GPU batch pricing for risk management.

---
**Report compiled for: Langtao Wang**  
**Date: April 3, 2026**
