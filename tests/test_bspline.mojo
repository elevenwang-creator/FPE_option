from numerics.bspline.knots import GenerateKnots
from numerics.bspline.basis import BSplineBasis
from numerics.bspline.recombination import RecombinationBasis
from numerics.bspline.tensor_product import TensorProductBasis

from std.testing import assert_true, assert_equal, TestSuite


def assert_float_close(a: Float64, b: Float64, atol: Float64 = 1e-10) raises:
    var diff = a - b
    if diff < 0:
        diff = -diff
    assert_true(diff < atol, "Expected " + String(b) + " got " + String(a))


def test_bspline_linear_values_degree_1() raises:
    var knots: List[Float64] = [0.0, 0.0, 0.5, 1.0, 1.0]
    var basis = BSplineBasis[1](knots^)

    var x1 = Float64(0.25)
    assert_float_close(basis.de_boor_cox(x1, 0), 0.5)
    assert_float_close(basis.de_boor_cox(x1, 1), 0.5)
    assert_float_close(basis.de_boor_cox(x1, 2), 0.0)

    var x2 = Float64(0.75)
    assert_float_close(basis.de_boor_cox(x2, 0), 0.0)
    assert_float_close(basis.de_boor_cox(x2, 1), 0.5)
    assert_float_close(basis.de_boor_cox(x2, 2), 0.5)


def test_partition_of_unity_degree_1() raises:
    var knots: List[Float64] = [0.0, 0.0, 0.5, 1.0, 1.0]
    var basis = BSplineBasis[1](knots^)

    var sample_points: List[Float64] = [0.1, 0.25, 0.5, 0.75, 0.9]
    for x in sample_points:
        var total = Float64(0.0)
        for i in range(basis.num_basis):
            total += basis.de_boor_cox(x, i)
        assert_float_close(total, 1.0)


def test_generate_knots_uniform_repeats_boundaries() raises:
    var gen = GenerateKnots(
        n=8,
        degree=2,
        method="uniform",
        center=0.2,
        boundary=(0.0, 1.0),
        mean=50.0,
        std=0.1,
    )

    var knots = gen.generate_knots()
    assert_equal(len(knots), 8)
    assert_float_close(knots[0], 0.0)
    assert_float_close(knots[1], 0.0)
    assert_float_close(knots[6], 1.0)
    assert_float_close(knots[7], 1.0)


def test_recombination_dirichlet_dirichlet_shape() raises:
    var knots: List[Float64] = [0.0, 0.0, 0.5, 1.0, 1.0]
    var rb = RecombinationBasis[1](
        basis=BSplineBasis[1](knots^),
        left_cond="dirichlet",
        right_cond="dirichlet",
    )

    var R = rb.recombination_matrix()
    assert_equal(R.nrows, 3)
    assert_equal(R.ncols, 1)


def test_recombination_neumann_neumann_shape() raises:
    var knots: List[Float64] = [0.0, 0.0, 0.5, 1.0, 1.0]
    var rb = RecombinationBasis[1](
        basis=BSplineBasis[1](knots^),
        left_cond="neumann",
        right_cond="neumann",
    )

    var R = rb.recombination_matrix()
    assert_equal(R.nrows, 3)
    assert_equal(R.ncols, 1)
    assert_equal(R.nnz(), 3)


def test_recombination_all_four_conditions() raises:
    var knots: List[Float64] = [0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0]
    var basis = BSplineBasis[1](knots.copy())
    var n = basis.num_basis

    var rb_dd = RecombinationBasis[1](
        basis=basis.copy(),
        left_cond="dirichlet",
        right_cond="dirichlet",
    )
    var R_dd = rb_dd.recombination_matrix()
    assert_equal(R_dd.nrows, n)
    assert_equal(R_dd.ncols, n - 2)
    assert_equal(R_dd.nnz(), n - 2)

    var rb_dn = RecombinationBasis[1](
        basis=basis.copy(),
        left_cond="dirichlet",
        right_cond="neumann",
    )
    var R_dn = rb_dn.recombination_matrix()
    assert_equal(R_dn.nrows, n)
    assert_equal(R_dn.ncols, n - 2)
    assert_equal(R_dn.nnz(), n - 1)

    var rb_nd = RecombinationBasis[1](
        basis=basis.copy(),
        left_cond="neumann",
        right_cond="dirichlet",
    )
    var R_nd = rb_nd.recombination_matrix()
    assert_equal(R_nd.nrows, n)
    assert_equal(R_nd.ncols, n - 2)
    assert_equal(R_nd.nnz(), n - 1)

    var rb_nn = RecombinationBasis[1](
        basis=basis.copy(),
        left_cond="neumann",
        right_cond="neumann",
    )
    var R_nn = rb_nn.recombination_matrix()
    assert_equal(R_nn.nrows, n)
    assert_equal(R_nn.ncols, n - 2)
    assert_equal(R_nn.nnz(), n)


def test_tensor_product_shapes() raises:
    var knots: List[Float64] = [0.0, 0.0, 0.5, 1.0, 1.0]
    var bs = RecombinationBasis[1](
        basis=BSplineBasis[1](knots.copy()),
        left_cond="neumann",
        right_cond="neumann",
    )
    var bv = RecombinationBasis[1](
        basis=BSplineBasis[1](knots.copy()),
        left_cond="neumann",
        right_cond="neumann",
    )
    var tp = TensorProductBasis[1, 1](basis_s=bs^, basis_v=bv^)

    var s_points: List[Float64] = [0.25, 0.75]
    var v_points: List[Float64] = [0.25, 0.75]
    var B = tp.eval_tensor(s_points, v_points)

    assert_equal(B.nrows, 4)
    assert_equal(B.ncols, 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
