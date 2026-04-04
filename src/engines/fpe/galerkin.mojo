"""Galerkin assembler for FPE mass and stiffness matrices.

Key improvement: uses native sparse add/scale/transpose from sparse.ops
instead of O(n²) dense round-trips. Assembly complexity is now O(nnz)
for add/scale operations instead of O(n²).
"""

from engines.fpe.domain import FPEDomain
from engines.fpe.heston_params import HestonParams
from numerics.utils import zeros_mat
from sparse.csr import CSRMatrix
from sparse.ops import add, kron, scale, spgemm
from std.algorithm import parallelize


def _identity(n: Int) -> CSRMatrix[DType.float64]:
    """Identity matrix as sparse CSR."""
    var dense = zeros_mat(n, n)
    for i in range(n):
        dense[i][i] = 1.0
    return CSRMatrix[DType.float64].from_dense(dense)


def _diag(values: List[Float64]) -> CSRMatrix[DType.float64]:
    """Diagonal matrix from values as sparse CSR."""
    var n = len(values)
    var dense = zeros_mat(n, n)
    for i in range(n):
        dense[i][i] = values[i]
    return CSRMatrix[DType.float64].from_dense(dense)


def _diag_left_mul(
    Phi: CSRMatrix[DType.float64], weights: List[Float64]
) -> CSRMatrix[DType.float64]:
    """Left-multiply by diagonal weight matrix: diag(w) @ Phi.

    Operates directly on sparse data without dense conversion.
    """
    var out = CSRMatrix[DType.float64](Phi.nrows, Phi.ncols)
    out.indptr = Phi.indptr.copy()
    out.indices = Phi.indices.copy()
    out.data = []
    for i in range(Phi.nrows):
        var w = 0.0
        if i < len(weights):
            w = weights[i]
        var row_start = Phi.indptr[i]
        var row_end = Phi.indptr[i + 1]
        for p in range(row_start, row_end):
            out.data.append(w * Phi.data[p])
    return out^


def _build_weight_vector(domain: FPEDomain) -> List[Float64]:
    """Build quadrature weight vector for S×V grid."""
    var weights: List[Float64] = []
    var jacobian = domain.jacobian_factor()
    for i in range(len(domain.s_weights)):
        for j in range(len(domain.v_weights)):
            weights.append(domain.s_weights[i] * domain.v_weights[j] * jacobian)
    return weights^


def _spT_diag_sp(
    Phi: CSRMatrix[DType.float64], weights: List[Float64]
) -> CSRMatrix[DType.float64]:
    """Compute Φ^T @ diag(w) @ Φ using sparse operations."""
    var left = Phi.transpose()  # O(nnz) sparse transpose
    var right = _diag_left_mul(Phi, weights)
    return spgemm(left, right)


struct GalerkinAssembler[B: Int]:
    """Galerkin assembler for FPE mass and stiffness matrices.

    Batch parameter B is plumbed for future batch-aware assembly.
    """

    def __init__(out self):
        pass

    def mass_matrix(self, domain: FPEDomain) -> CSRMatrix[DType.float64]:
        """Assemble mass matrix M = Φ^T W Φ."""
        var basis = domain.build_basis()
        var Phi = basis.eval_tensor(domain.s_points, domain.v_points)
        var w = _build_weight_vector(domain)
        return _spT_diag_sp(Phi, w)

    def stiffness_matrix(
        self, domain: FPEDomain, params: HestonParams
    ) -> CSRMatrix[DType.float64]:
        """Assemble stiffness matrix K for the Heston FPE.

        Uses native sparse add/scale throughout — no dense round-trips.
        """
        var basis = domain.build_basis()

        var Phi = basis.eval_tensor(domain.s_points, domain.v_points)
        var s_partial = basis.partial_s(domain.s_points, domain.v_points)
        var v_partial = basis.partial_v(domain.s_points, domain.v_points)

        var n_s = len(domain.s_points)
        var n_v = len(domain.v_points)

        var W = _diag(_build_weight_vector(domain))

        var I_s = _identity(n_s)
        var I_v = _identity(n_v)

        var s_diag = kron(_diag(domain.s_points), I_v)
        var v_diag = kron(I_s, _diag(domain.v_points))

        var s_sq_vals: List[Float64] = []
        for i in range(n_s):
            s_sq_vals.append(domain.s_points[i] * domain.s_points[i])
        var s_sq_diag = kron(_diag(s_sq_vals), I_v)

        var sv_diag = spgemm(s_diag, v_diag)
        var ssv_diag = spgemm(s_sq_diag, v_diag)

        var j = domain.jacobian_factor()
        if j == 0.0:
            j = 1.0

        var k1 = (-params.r + 0.5 * params.rho * params.sigma) / j
        var k2 = 1.0 / j
        var k3 = 0.5 / (j * j)
        var k4 = 0.5 * params.rho * params.sigma / j
        var k5 = 0.5 * params.sigma * params.sigma - params.kappa * params.theta
        var k6 = params.kappa + 0.5 * params.rho * params.sigma
        var k8 = 0.5 * params.sigma * params.sigma

        var s_diag_W = spgemm(s_diag, W)
        var sv_diag_W = spgemm(sv_diag, W)
        var ssv_diag_W = spgemm(ssv_diag, W)
        var v_diag_W = spgemm(v_diag, W)

        # Native sparse add/scale — O(nnz) instead of O(n²) dense round-trips
        var integ_sbw = add(scale(k1, s_diag_W), scale(k2, sv_diag_W))
        var integ_svw = scale(k4, sv_diag_W)
        var integ_ssvw = scale(k3, ssv_diag_W)

        var integ_vbw = add(scale(k5, W), scale(k6, v_diag_W))
        var integ_vw = scale(k8, v_diag_W)

        var s_partial_T = s_partial.transpose()
        var v_partial_T = v_partial.transpose()

        var K_sb = spgemm(spgemm(s_partial_T, integ_sbw), Phi)
        var K_sv = spgemm(spgemm(s_partial_T, integ_svw), v_partial)
        var K_ssv = spgemm(spgemm(s_partial_T, integ_ssvw), s_partial)

        var K_vb = spgemm(spgemm(v_partial_T, integ_vbw), Phi)
        var K_vv = spgemm(spgemm(v_partial_T, integ_vw), v_partial)
        var K_vs = K_sv.transpose()

        return add(add(add(K_sb, K_vb), add(K_ssv, K_vv)), add(K_sv, K_vs))
