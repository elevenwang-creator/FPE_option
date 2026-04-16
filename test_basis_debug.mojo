from numerics.bspline.knots import GenerateKnots
from numerics.bspline.basis import BSplineBasis
from numerics.bspline.recombination import RecombinationBasis
from sparse.csr import CSRMatrix

def main():
    var gen_s = GenerateKnots(18, 3, "chebyshev", 0.2, (0.0, 1.0))
    var knots_s = gen_s.generate_knots()
    var basis_s = BSplineBasis[3](knots_s.copy())
    var recomb_s = RecombinationBasis[3](basis_s.copy(), "dirichlet", "neumann")

    print("num_basis=" + String(basis_s.num_basis))
    print("knots len=" + String(len(knots_s)))

    var R = recomb_s.recombination_matrix()
    print("R: " + String(R.nrows) + "x" + String(R.ncols) + " nnz=" + String(R.nnz()))
    for i in range(R.nrows):
        for p in range(R.indptr[i], R.indptr[i + 1]):
            print("  R[" + String(i) + "," + String(R.indices[p]) + "]=" + String(R.data[p]))

    var test_pts: List[Float64] = [0.1, 0.3, 0.5, 0.7, 0.9]

    var B_raw = basis_s.eval_all(test_pts)
    print("B_raw: " + String(B_raw.nrows) + "x" + String(B_raw.ncols) + " nnz=" + String(B_raw.nnz()))
    for i in range(B_raw.nrows):
        for p in range(B_raw.indptr[i], B_raw.indptr[i + 1]):
            print("  B[" + String(i) + "," + String(B_raw.indices[p]) + "]=" + String(B_raw.data[p]))

    var B_recomb = recomb_s.eval_all(test_pts)
    print("B_recomb: " + String(B_recomb.nrows) + "x" + String(B_recomb.ncols) + " nnz=" + String(B_recomb.nnz()))
    for i in range(B_recomb.nrows):
        for p in range(B_recomb.indptr[i], B_recomb.indptr[i + 1]):
            print("  Br[" + String(i) + "," + String(B_recomb.indices[p]) + "]=" + String(B_recomb.data[p]))
