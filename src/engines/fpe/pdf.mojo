"""PDF computation from FPE solution coefficients.

Uses Kronecker-structured spmv: kron(Bs, Bv).spmv(q) = vec(Bs @ Q @ Bv^T)
operates on small 1D factors instead of 2.1M-row kron matrix.
"""

from engines.fpe.domain import FPECachedBasis
from sparse.kron_spmv import kron_spmv


def _reshape_to_grid(
    flat: List[Float64], n_s: Int, n_v: Int
) -> List[List[Float64]]:
    var out: List[List[Float64]] = []
    var idx = 0
    for _ in range(n_s):
        var row: List[Float64] = []
        for _ in range(n_v):
            var value = 0.0
            if idx < len(flat):
                value = flat[idx]
            row.append(value)
            idx += 1
        out.append(row^)
    return out^


def pdf_from_cached[ds: Int, dv: Int](
    cached: FPECachedBasis[ds, dv], q_t: List[Float64]
) -> List[List[Float64]]:
    var pdf_flat = kron_spmv(cached.Bs, cached.Bv, q_t)
    return _reshape_to_grid(pdf_flat, cached.n_s, cached.n_v)
