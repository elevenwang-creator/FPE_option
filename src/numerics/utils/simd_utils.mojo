from std.sys import simd_width_of
from std.algorithm.backend.vectorize import vectorize
from layout import TileTensor
from layout.tile_layout import TensorLayout

comptime SIMD_W: Int = simd_width_of[DType.float64]()


def vec_scale(src: List[Float64], alpha: Float64) -> List[Float64]:
    var n = len(src)
    var out = List[Float64](length=n, fill=0.0)
    var s_ptr = src.unsafe_ptr()
    var r_ptr = out.unsafe_ptr()

    def vscale[width: Int](p_off: Int) {read alpha, read s_ptr, read r_ptr}:
        for w in range(width):
            r_ptr[p_off + w] = alpha * s_ptr[p_off + w]

    vectorize[SIMD_W](n, vscale)
    return out^


def vec_central_diff(
    p_up: List[Float64], p_dn: List[Float64], h: Float64
) -> List[Float64]:
    var n = len(p_up)
    var out = List[Float64](length=n, fill=0.0)
    var up_ptr = p_up.unsafe_ptr()
    var dn_ptr = p_dn.unsafe_ptr()
    var r_ptr = out.unsafe_ptr()
    var inv_2h = 1.0 / (2.0 * h)

    def vcd[width: Int](p_off: Int) {read inv_2h, read up_ptr, read dn_ptr, read r_ptr}:
        for w in range(width):
            r_ptr[p_off + w] = (up_ptr[p_off + w] - dn_ptr[p_off + w]) * inv_2h

    vectorize[SIMD_W](n, vcd)
    return out^


def vec_second_diff(
    p_up: List[Float64], p_base: List[Float64], p_dn: List[Float64], h: Float64
) -> List[Float64]:
    var n = len(p_up)
    var out = List[Float64](length=n, fill=0.0)
    var up_ptr = p_up.unsafe_ptr()
    var base_ptr = p_base.unsafe_ptr()
    var dn_ptr = p_dn.unsafe_ptr()
    var r_ptr = out.unsafe_ptr()
    var inv_h2 = 1.0 / (h * h)

    def vsd[width: Int](p_off: Int) {read inv_h2, read up_ptr, read base_ptr, read dn_ptr, read r_ptr}:
        for w in range(width):
            r_ptr[p_off + w] = (
                up_ptr[p_off + w]
                - 2.0 * base_ptr[p_off + w]
                + dn_ptr[p_off + w]
            ) * inv_h2

    vectorize[SIMD_W](n, vsd)
    return out^


def mat_vec_mul(
    mat: TileTensor[DType.float64, ...],
    x: Span[Float64, ...],
    mut y: Span[mut=True, Float64, ...],
):
    var m = Int(mat.dim[0]())
    var n = Int(mat.dim[1]())
    var x_ptr = x.unsafe_ptr()
    for i in range(m):
        var dot: Float64 = 0.0
        var base = i * n
        def vdot_row[width: Int](p_off: Int) {mut dot, read mat, read x_ptr, read base}:
            dot += (
                mat.raw_load[width=width](base + p_off) *
                x_ptr.load[width=width](p_off)
            ).reduce_add()
        vectorize[SIMD_W](n, vdot_row)
        y[i] = dot


def mat_mul(
    A: TileTensor[DType.float64, ...],
    B: TileTensor[DType.float64, ...],
    mut C: TileTensor[mut=True, DType.float64, ...],
):
    var m = Int(A.dim[0]())
    var n = Int(A.dim[1]())
    var k = Int(B.dim[1]())

    for i in range(m):
        var row_a = i * n
        var row_c = i * k
        for p in range(n):
            var a_ip = A.raw_load[width=1](row_a + p)[0]
            if a_ip == 0.0:
                continue
            var base_b = p * k
            def vaxpy[width: Int](j_off: Int) {mut C, read a_ip, read B, read base_b, read row_c}:
                var bv = B.raw_load[width=width](base_b + j_off)
                var cv = C.raw_load[width=width](row_c + j_off)
                C.raw_store[width=width](row_c + j_off, cv + a_ip * bv)
            vectorize[SIMD_W](k, vaxpy)
