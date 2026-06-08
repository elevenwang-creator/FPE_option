# Third-Party Notices

This project incorporates or references the following third-party software:

## RADAU5 (Fortran ODE Solver)

**Source:** Ernst Hairer and Gerhard Wanner  
**Reference:** *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*, Springer, 1996.  
**Location:** `benchmarks/radau5.f`, `benchmarks/dc_lapack.f`, and related `.f` files under `benchmarks/`

RADAU5 is an implicit Runge-Kutta method of order 5 (Radau IIA) with step size control and continuous output. These Fortran reference implementations are used for cross-verification and benchmarking purposes only and are not part of the distributed library.

## Python Reference Implementation

**Source:** Stoykov, S. (2024). *Numerical Solution of Fokker-Planck Equation by Variational Approach -- an Application to Pricing Barrier Options.* Wilmott, 2024(133).  
**Location:** `debug_python_ref.py`, `docs/python_reference/`

The original Python FPE solver from which this engine was ported. Used for cross-verification and benchmarking.
