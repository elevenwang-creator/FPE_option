"""Benchmark: Mojo NDArray Array Operations with SIMD

Compares Mojo NDArray SIMD performance with:
- C++ std::vector (see bench_ndarray_cpp.cpp)
- Python NumPy (see bench_ndarray.py)

Operations tested with SIMD:
- Element-wise addition (add_from)
- Scalar multiplication (scale_assign)
- Dot product (SIMD reduction)
- Vector sum (SIMD reduction)
- Sqrt (element-wise with SIMD)
"""

from numerics.utils.ndarray import NDArray
from std.algorithm import parallelize
from std.benchmark import run as bench_run
from std.benchmark import Unit
from std.math import sqrt
from std.sys import simd_width_of


struct ArrayBenchmark:
    var size: Int
    var a: NDArray[Float64]
    var b: NDArray[Float64]
    var c: NDArray[Float64]
    var sink: Float64

    def __init__(out self, size: Int):
        self.size = size
        self.a = NDArray[Float64](size)
        self.b = NDArray[Float64](size)
        self.c = NDArray[Float64](size)
        self.sink = 0.0
        for i in range(size):
            self.a[i] = Float64(i) * 0.1
            self.b[i] = Float64(i) * 0.2 + 1.0

    def bench_element_wise_add_simd(mut self):
        self.c.add_from(self.a, self.b)

    def bench_scalar_mul_simd(mut self):
        self.a.scale_assign(3.14159)

    def bench_dot_product_simd(mut self) -> Float64:
        var p = self.a._f64_ptr()
        var pb = self.b._f64_ptr()
        var n = self.size
        comptime width = simd_width_of[DType.float64]()
        var total = SIMD[DType.float64, width](0.0)
        var i = 0
        while i + width <= n:
            var sa = (p + i).load[width=width]()
            var sb = (pb + i).load[width=width]()
            total += sa * sb
            i += width
        var result = total.reduce_add()
        while i < n:
            result += p[i] * pb[i]
            i += 1
        self.sink = result
        return result

    def bench_vector_sum_simd(mut self) -> Float64:
        var p = self.a._f64_ptr()
        var n = self.size
        comptime width = simd_width_of[DType.float64]()
        var total = SIMD[DType.float64, width](0.0)
        var i = 0
        while i + width <= n:
            var sa = (p + i).load[width=width]()
            total += sa
            i += width
        var result = total.reduce_add()
        while i < n:
            result += p[i]
            i += 1
        self.sink = result
        return result

    def bench_sqrt_simd(mut self):
        var p = self.a._f64_ptr()
        var n = self.size
        comptime width = simd_width_of[DType.float64]()
        var i = 0
        while i + width <= n:
            var sv = (p + i).load[width=width]()
            (p + i).store[width=width](sqrt(sv))
            i += width
        while i < n:
            p[i] = sqrt(p[i])
            i += 1

    def bench_dot_product_parallel_simd(mut self) -> Float64:
        var pa = self.a._f64_ptr()
        var pb = self.b._f64_ptr()
        var n = self.size
        var num_threads = 8
        var chunk_size = (n + num_threads - 1) // num_threads
        var partials: List[Float64] = []
        for _ in range(num_threads):
            partials.append(0.0)

        @parameter
        def worker(i: Int):
            var tid = i
            var start = tid * chunk_size
            var end = start + chunk_size
            if end > n:
                end = n
            var total = SIMD[DType.float64, 8](0.0)
            var j = start
            while j + 8 <= end:
                var sa = (pa + j).load[width=8]()
                var sb = (pb + j).load[width=8]()
                total += sa * sb
                j += 8
            var result = total.reduce_add()
            while j < end:
                result += pa[j] * pb[j]
                j += 1
            partials[tid] = result

        parallelize[worker](num_threads, num_threads)
        var final_result: Float64 = 0.0
        for i in range(num_threads):
            final_result += partials[i]
        self.sink = final_result
        return final_result

    def bench_vector_sum_parallel_simd(mut self) -> Float64:
        var pa = self.a._f64_ptr()
        var n = self.size
        var num_threads = 8
        var chunk_size = (n + num_threads - 1) // num_threads
        var partials: List[Float64] = []
        for _ in range(num_threads):
            partials.append(0.0)

        @parameter
        def worker(i: Int):
            var tid = i
            var start = tid * chunk_size
            var end = start + chunk_size
            if end > n:
                end = n
            var total = SIMD[DType.float64, 8](0.0)
            var j = start
            while j + 8 <= end:
                var sa = (pa + j).load[width=8]()
                total += sa
                j += 8
            var result = total.reduce_add()
            while j < end:
                result += pa[j]
                j += 1
            partials[tid] = result

        parallelize[worker](num_threads, num_threads)
        var final_result: Float64 = 0.0
        for i in range(num_threads):
            final_result += partials[i]
        self.sink = final_result
        return final_result

    def bench_dot_product_comptime_simd(mut self) -> Float64:
        var pa = self.a._f64_ptr()
        var pb = self.b._f64_ptr()
        var n = self.size
        var total = SIMD[DType.float64, 8](0.0)
        var i = 0
        while i + 64 <= n:
            comptime for k in range(8):
                var sa = (pa + i + k*8).load[width=8]()
                var sb = (pb + i + k*8).load[width=8]()
                total += sa * sb
            i += 64
        while i + 8 <= n:
            var sa = (pa + i).load[width=8]()
            var sb = (pb + i).load[width=8]()
            total += sa * sb
            i += 8
        var result = total.reduce_add()
        while i < n:
            result += pa[i] * pb[i]
            i += 1
        self.sink = result
        return result

    def bench_vector_sum_comptime_simd(mut self) -> Float64:
        var pa = self.a._f64_ptr()
        var n = self.size
        var total = SIMD[DType.float64, 8](0.0)
        var i = 0
        while i + 64 <= n:
            comptime for k in range(8):
                var sa = (pa + i + k*8).load[width=8]()
                total += sa
            i += 64
        while i + 8 <= n:
            var sa = (pa + i).load[width=8]()
            total += sa
            i += 8
        var result = total.reduce_add()
        while i < n:
            result += pa[i]
            i += 1
        self.sink = result
        return result

    def bench_element_wise_add_comptime_if_simd(mut self):
        var pa = self.a._f64_ptr()
        var pb = self.b._f64_ptr()
        var pc = self.c._f64_ptr()
        var n = self.size
        var i = 0
        comptime if simd_width_of[DType.float64]() == 8:
            while i + 8 <= n:
                var sa = (pa + i).load[width=8]()
                var sb = (pb + i).load[width=8]()
                (pc + i).store[width=8](sa + sb)
                i += 8
        else:
            while i + 2 <= n:
                var sa = (pa + i).load[width=2]()
                var sb = (pb + i).load[width=2]()
                (pc + i).store[width=2](sa + sb)
                i += 2
        while i < n:
            pc[i] = pa[i] + pb[i]
            i += 1


struct ListBenchmark:
    var size: Int
    var a: List[Float64]
    var b: List[Float64]
    var c: List[Float64]
    var sink: Float64

    def __init__(out self, size: Int):
        self.size = size
        self.a = List[Float64]()
        self.b = List[Float64]()
        self.c = List[Float64]()
        self.sink = 0.0
        for i in range(size):
            self.a.append(Float64(i) * 0.1)
            self.b.append(Float64(i) * 0.2 + 1.0)
            self.c.append(0.0)

    def bench_element_wise_add(mut self):
        for i in range(self.size):
            self.c[i] = self.a[i] + self.b[i]

    def bench_scalar_mul(mut self):
        for i in range(self.size):
            self.a[i] = self.a[i] * 3.14159

    @no_inline
    def bench_dot_product(mut self) -> Float64:
        var result: Float64 = 0.0
        for i in range(self.size):
            result += self.a[i] * self.b[i]
        self.sink = result
        return result

    @no_inline
    def bench_vector_sum(mut self) -> Float64:
        var result: Float64 = 0.0
        for i in range(self.size):
            result += self.a[i]
        self.sink = result
        return result

    def bench_sqrt(mut self):
        for i in range(self.size):
            self.a[i] = sqrt(self.a[i])

    def bench_element_wise_add_parallel(mut self):
        var pa = self.a._data
        var pb = self.b._data
        var pc = self.c._data
        var n = self.size

        @parameter
        def worker(i: Int):
            pc[i] = pa[i] + pb[i]

        parallelize[worker](n)

    def bench_scalar_mul_parallel(mut self):
        var pa = self.a._data
        var n = self.size
        var alpha = 3.14159

        @parameter
        def worker(i: Int):
            pa[i] = pa[i] * alpha

        parallelize[worker](n)

    @no_inline
    def bench_dot_product_parallel(mut self) -> Float64:
        var pa = self.a._data
        var pb = self.b._data
        var n = self.size
        var local_sink = 0.0

        @parameter
        def worker(i: Int):
            local_sink += pa[i] * pb[i]

        parallelize[worker](n)
        self.sink = local_sink
        return local_sink

    @no_inline
    def bench_vector_sum_parallel(mut self) -> Float64:
        var pa = self.a._data
        var n = self.size
        var local_sink = 0.0

        @parameter
        def worker(i: Int):
            local_sink += pa[i]

        parallelize[worker](n)
        self.sink = local_sink
        return local_sink

    def bench_sqrt_parallel(mut self):
        var pa = self.a._data

        @parameter
        def worker(i: Int):
            pa[i] = sqrt(pa[i])

        parallelize[worker](self.size)


def run_mojo_benchmark[size: Int]() raises:
    var bench = ArrayBenchmark(size)
    var list_bench = ListBenchmark(size)

    print("  [Mojo NDArray SIMD] Size:", size)
    print()

    def bench_add() capturing:
        bench.bench_element_wise_add_simd()

    var report_add = bench_run[bench_add](
        num_warmup_iters=3,
        max_iters=100,
        min_runtime_secs=0.5,
    )
    print("  Element-wise add (SIMD add_from):")
    print("    Mean:", report_add.mean(Unit.ms), "ms")
    print("    Min: ", report_add.min(Unit.ms), "ms")
    print()

    def bench_mul() capturing:
        bench.bench_scalar_mul_simd()

    var report_mul = bench_run[bench_mul](
        num_warmup_iters=3,
        max_iters=100,
        min_runtime_secs=0.5,
    )
    print("  Scalar multiply (SIMD scale_assign):")
    print("    Mean:", report_mul.mean(Unit.ms), "ms")
    print("    Min: ", report_mul.min(Unit.ms), "ms")
    print()

    def bench_dot() capturing:
        _ = bench.bench_dot_product_simd()

    var report_dot = bench_run[bench_dot](
        num_warmup_iters=3,
        max_iters=100,
        min_runtime_secs=0.5,
    )
    print("  Dot product (SIMD reduction):")
    print("    Mean:", report_dot.mean(Unit.ms), "ms")
    print("    Min: ", report_dot.min(Unit.ms), "ms")
    print()

    def bench_sum() capturing:
        _ = bench.bench_vector_sum_simd()

    var report_sum = bench_run[bench_sum](
        num_warmup_iters=3,
        max_iters=100,
        min_runtime_secs=0.5,
    )
    print("  Vector sum (SIMD reduction):")
    print("    Mean:", report_sum.mean(Unit.ms), "ms")
    print("    Min: ", report_sum.min(Unit.ms), "ms")
    print()

    def bench_sqrt_op() capturing:
        bench.bench_sqrt_simd()

    var report_sqrt = bench_run[bench_sqrt_op](
        num_warmup_iters=3,
        max_iters=100,
        min_runtime_secs=0.5,
    )
    print("  Sqrt (SIMD):")
    print("    Mean:", report_sqrt.mean(Unit.ms), "ms")
    print("    Min: ", report_sqrt.min(Unit.ms), "ms")
    print()

    print("  [Mojo NDArray Parallel SIMD] Size:", size)
    print()

    def bench_dot_par() capturing:
        _ = bench.bench_dot_product_parallel_simd()

    var report_dot_par = bench_run[bench_dot_par](
        num_warmup_iters=3,
        max_iters=50,
        min_runtime_secs=0.5,
    )
    print("  Dot product (Parallel SIMD 8-lane):")
    print("    Mean:", report_dot_par.mean(Unit.ms), "ms")
    print("    Min: ", report_dot_par.min(Unit.ms), "ms")
    print()

    def bench_sum_par() capturing:
        _ = bench.bench_vector_sum_parallel_simd()

    var report_sum_par = bench_run[bench_sum_par](
        num_warmup_iters=3,
        max_iters=50,
        min_runtime_secs=0.5,
    )
    print("  Vector sum (Parallel SIMD 8-lane):")
    print("    Mean:", report_sum_par.mean(Unit.ms), "ms")
    print("    Min: ", report_sum_par.min(Unit.ms), "ms")
    print()

    print("  [Mojo Comptime SIMD] Size:", size)
    print()

    def bench_dot_comptime() capturing:
        _ = bench.bench_dot_product_comptime_simd()

    var report_dot_comptime = bench_run[bench_dot_comptime](
        num_warmup_iters=3,
        max_iters=100,
        min_runtime_secs=0.5,
    )
    print("  Dot product (Comptime for SIMD):")
    print("    Mean:", report_dot_comptime.mean(Unit.ms), "ms")
    print("    Min: ", report_dot_comptime.min(Unit.ms), "ms")
    print()

    def bench_sum_comptime() capturing:
        _ = bench.bench_vector_sum_comptime_simd()

    var report_sum_comptime = bench_run[bench_sum_comptime](
        num_warmup_iters=3,
        max_iters=100,
        min_runtime_secs=0.5,
    )
    print("  Vector sum (Comptime for SIMD):")
    print("    Mean:", report_sum_comptime.mean(Unit.ms), "ms")
    print("    Min: ", report_sum_comptime.min(Unit.ms), "ms")
    print()

    def bench_add_comptime() capturing:
        bench.bench_element_wise_add_comptime_if_simd()

    var report_add_comptime = bench_run[bench_add_comptime](
        num_warmup_iters=3,
        max_iters=100,
        min_runtime_secs=0.5,
    )
    print("  Element-wise add (Comptime if SIMD):")
    print("    Mean:", report_add_comptime.mean(Unit.ms), "ms")
    print("    Min: ", report_add_comptime.min(Unit.ms), "ms")
    print()

    print("  [Mojo List] Size:", size)
    print()

    def bench_list_add() capturing:
        list_bench.bench_element_wise_add()

    var report_list_add = bench_run[bench_list_add](
        num_warmup_iters=3,
        max_iters=100,
        min_runtime_secs=0.5,
    )
    print("  Element-wise add (List):")
    print("    Mean:", report_list_add.mean(Unit.ms), "ms")
    print("    Min: ", report_list_add.min(Unit.ms), "ms")
    print()

    def bench_list_mul() capturing:
        list_bench.bench_scalar_mul()

    var report_list_mul = bench_run[bench_list_mul](
        num_warmup_iters=3,
        max_iters=100,
        min_runtime_secs=0.5,
    )
    print("  Scalar multiply (List):")
    print("    Mean:", report_list_mul.mean(Unit.ms), "ms")
    print("    Min: ", report_list_mul.min(Unit.ms), "ms")
    print()

    def bench_list_dot() capturing:
        _ = list_bench.bench_dot_product()

    var report_list_dot = bench_run[bench_list_dot](
        num_warmup_iters=3,
        max_iters=100,
        min_runtime_secs=0.5,
    )
    print("  Dot product (List):")
    print("    Mean:", report_list_dot.mean(Unit.ms), "ms")
    print("    Min: ", report_list_dot.min(Unit.ms), "ms")
    print()

    def bench_list_sum() capturing:
        _ = list_bench.bench_vector_sum()

    var report_list_sum = bench_run[bench_list_sum](
        num_warmup_iters=3,
        max_iters=100,
        min_runtime_secs=0.5,
    )
    print("  Vector sum (List):")
    print("    Mean:", report_list_sum.mean(Unit.ms), "ms")
    print("    Min: ", report_list_sum.min(Unit.ms), "ms")
    print()

    def bench_list_sqrt() capturing:
        list_bench.bench_sqrt()

    var report_list_sqrt = bench_run[bench_list_sqrt](
        num_warmup_iters=3,
        max_iters=100,
        min_runtime_secs=0.5,
    )
    print("  Sqrt (List):")
    print("    Mean:", report_list_sqrt.mean(Unit.ms), "ms")
    print("    Min: ", report_list_sqrt.min(Unit.ms), "ms")
    print()

    def bench_list_add_par() capturing:
        list_bench.bench_element_wise_add_parallel()

    var report_list_add_par = bench_run[bench_list_add_par](
        num_warmup_iters=3,
        max_iters=100,
        min_runtime_secs=0.5,
    )
    print("  Element-wise add (List parallel):")
    print("    Mean:", report_list_add_par.mean(Unit.ms), "ms")
    print("    Min: ", report_list_add_par.min(Unit.ms), "ms")
    print()

    def bench_list_mul_par() capturing:
        list_bench.bench_scalar_mul_parallel()

    var report_list_mul_par = bench_run[bench_list_mul_par](
        num_warmup_iters=3,
        max_iters=100,
        min_runtime_secs=0.5,
    )
    print("  Scalar multiply (List parallel):")
    print("    Mean:", report_list_mul_par.mean(Unit.ms), "ms")
    print("    Min: ", report_list_mul_par.min(Unit.ms), "ms")
    print()

    def bench_list_dot_par() capturing:
        _ = list_bench.bench_dot_product_parallel()

    var report_list_dot_par = bench_run[bench_list_dot_par](
        num_warmup_iters=3,
        max_iters=100,
        min_runtime_secs=0.5,
    )
    print("  Dot product (List parallel):")
    print("    Mean:", report_list_dot_par.mean(Unit.ms), "ms")
    print("    Min: ", report_list_dot_par.min(Unit.ms), "ms")
    print()

    def bench_list_sum_par() capturing:
        _ = list_bench.bench_vector_sum_parallel()

    var report_list_sum_par = bench_run[bench_list_sum_par](
        num_warmup_iters=3,
        max_iters=100,
        min_runtime_secs=0.5,
    )
    print("  Vector sum (List parallel):")
    print("    Mean:", report_list_sum_par.mean(Unit.ms), "ms")
    print("    Min: ", report_list_sum_par.min(Unit.ms), "ms")
    print()

    def bench_list_sqrt_par() capturing:
        list_bench.bench_sqrt_parallel()

    var report_list_sqrt_par = bench_run[bench_list_sqrt_par](
        num_warmup_iters=3,
        max_iters=100,
        min_runtime_secs=0.5,
    )
    print("  Sqrt (List parallel):")
    print("    Mean:", report_list_sqrt_par.mean(Unit.ms), "ms")
    print("    Min: ", report_list_sqrt_par.min(Unit.ms), "ms")
    print()


def main() raises:
    print("=" * 70)
    print("  Mojo NDArray SIMD Benchmark")
    print("=" * 70)
    print()
    print("SIMD width:", simd_width_of[DType.float64](), "Float64 lanes")
    print("Operations: add, scale, dot, sum, sqrt")
    print()

    print("--- Small arrays (N=10000) ---")
    run_mojo_benchmark[10000]()

    print("--- Medium arrays (N=100000) ---")
    run_mojo_benchmark[100000]()

    print("--- Large arrays (N=1000000) ---")
    run_mojo_benchmark[1000000]()

    print("=" * 70)
    print("  Mojo NDArray SIMD Benchmark Complete")
    print("=" * 70)