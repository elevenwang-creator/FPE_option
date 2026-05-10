"""Sparse matrix operations — thin re-export hub.

All function bodies extracted to dedicated parallelized files:
- scale -> sparse.scale
- diag_scale -> sparse.diag_scale
- spgemm -> sparse.spgemm
- kron -> sparse.kron
- add -> sparse.add
- diag_mul -> sparse.diag_mul (diag_row_scale, diag_col_scale, diag_diag_mul)
- sparse_transpose -> DELETED (use CSRMatrix.transpose() method)

This module re-exports for backward compatibility only.
"""

from sparse.csr import CSRMatrix
from sparse.spgemm import spgemm
from sparse.kron import kron
from sparse.add import add
from sparse.scale import scale
from sparse.diag_scale import diag_scale
from sparse.diag_mul import diag_row_scale, diag_col_scale, diag_diag_mul
