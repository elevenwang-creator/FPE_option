#!/usr/bin/env python3
"""Benchmark: Python NumPy Array Operations

Compares Python NumPy performance with:
- Mojo NDArray (see bench_ndarray.mojo)
- C++ std::vector (see bench_ndarray_cpp.cpp)

Operations tested:
- Element-wise addition
- Scalar multiplication
- Dot product
- Vector sum
"""

import numpy as np
import time


def measure_ms(func, warmup=3, iterations=100, min_time=0.5):
    for _ in range(warmup):
        func()

    times = []
    elapsed = 0.0
    iters = 0

    while elapsed < min_time * 1000 or iters < iterations:
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)
        elapsed += times[-1]
        iters += 1
        if elapsed >= min_time * 1000 and iters >= iterations:
            break

    return np.mean(times)


class ArrayBenchmark:
    def __init__(self, size):
        self.size = size
        self.a = np.linspace(0, size * 0.1, size)
        self.b = np.linspace(1, size * 0.2 + 1, size)
        self.c = np.zeros(size)

    def bench_element_wise_add(self):
        self.c = self.a + self.b

    def bench_scalar_mul(self):
        self.a = self.a * 3.14159

    def bench_dot_product(self):
        return np.dot(self.a, self.b)

    def bench_vector_sum(self):
        return np.sum(self.a)

    def bench_sqrt(self):
        self.a = np.sqrt(self.a)


def run_benchmark(n):
    bench = ArrayBenchmark(n)

    print(f"  [Python NumPy] Size: {n}")
    print()

    t_add = measure_ms(bench.bench_element_wise_add)
    print("  Element-wise add:")
    print(f"    Mean: {t_add:.3f} ms")
    print()

    t_mul = measure_ms(bench.bench_scalar_mul)
    print("  Scalar multiply:")
    print(f"    Mean: {t_mul:.3f} ms")
    print()

    t_dot = measure_ms(lambda: bench.bench_dot_product())
    print("  Dot product:")
    print(f"    Mean: {t_dot:.3f} ms")
    print()

    t_sum = measure_ms(lambda: bench.bench_vector_sum())
    print("  Vector sum:")
    print(f"    Mean: {t_sum:.3f} ms")
    print()

    t_sqrt = measure_ms(bench.bench_sqrt)
    print("  Sqrt:")
    print(f"    Mean: {t_sqrt:.3f} ms")
    print()


def main():
    print("=" * 70)
    print("  Python NumPy Benchmark")
    print("=" * 70)
    print()
    print("Operations: element-wise add, scalar mul, dot product, sum, sqrt")
    print()

    print("--- Small arrays (N=10000) ---")
    run_benchmark(10000)

    print("--- Medium arrays (N=100000) ---")
    run_benchmark(100000)

    print("--- Large arrays (N=1000000) ---")
    run_benchmark(1000000)

    print("=" * 70)
    print("  Python NumPy Benchmark Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()