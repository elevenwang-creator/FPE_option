"""GPU Executor module for FPE.

Kernel implementations live alongside their CPU counterparts as _gpu.mojo files:
  - knots GPU     → numerics/bspline/knots_gpu.mojo
  - grid/basis/boundary GPU → engines/fpe/domain_gpu.mojo
  - SPmatrix GPU  → engines/fpe/galerkin_gpu.mojo
  - delta/initial GPU → engines/fpe/initial_cond_gpu.mojo
  - LU GPU        → numerics/linalg_gpu.mojo
  - RADAU5 GPU    → numerics/ode/radau_gpu.mojo
  - integrate GPU → engines/fpe/pdf_gpu.mojo
  - price integration GPU → engines/fpe/pdf_gpu.mojo
  - loss/LM GPU   → engines/calibrator/objective_gpu.mojo

This module only contains the orchestrator (GPUFullChainExecutor).
"""


