// Benchmark: C++ std::vector Array Operations
//
// Compares C++ std::vector performance with:
// - Mojo NDArray (see bench_ndarray.mojo)
// - Python NumPy (see bench_ndarray.py)
//
// Operations tested:
// - Element-wise addition
// - Scalar multiplication
// - Dot product
// - Vector sum

#include <vector>
#include <chrono>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <numeric>

template <typename Func>
double measure_ms(Func f, int warmup = 3, int iterations = 100, double min_time = 0.5) {
    for (int i = 0; i < warmup; ++i) {
        f();
    }

    auto start = std::chrono::high_resolution_clock::now();
    int iters = 0;
    double elapsed = 0.0;

    while (elapsed < min_time * 1000.0 || iters < iterations) {
        f();
        iters++;
        auto end = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        if (elapsed >= min_time * 1000.0 && iters >= iterations) break;
    }

    return elapsed / iters;
}

struct ArrayBenchmark {
    int size;
    std::vector<double> a;
    std::vector<double> b;
    std::vector<double> c;

    ArrayBenchmark(int n) : size(n), a(n), b(n), c(n) {
        for (int i = 0; i < size; ++i) {
            a[i] = double(i) * 0.1;
            b[i] = double(i) * 0.2 + 1.0;
        }
    }

    void bench_element_wise_add() {
        for (int i = 0; i < size; ++i) {
            c[i] = a[i] + b[i];
        }
    }

    void bench_scalar_mul() {
        for (int i = 0; i < size; ++i) {
            a[i] = a[i] * 3.14159;
        }
    }

    double bench_dot_product() {
        double result = 0.0;
        for (int i = 0; i < size; ++i) {
            result += a[i] * b[i];
        }
        return result;
    }

    double bench_vector_sum() {
        double result = 0.0;
        for (int i = 0; i < size; ++i) {
            result += a[i];
        }
        return result;
    }

    void bench_sqrt() {
        for (int i = 0; i < size; ++i) {
            a[i] = std::sqrt(a[i]);
        }
    }
};

template <int N>
void run_benchmark() {
    ArrayBenchmark bench(N);
    volatile double sink = 0.0;

    std::cout << "  [C++ std::vector] Size: " << N << std::endl;
    std::cout << std::endl;

    double t_add = measure_ms([&]() { bench.bench_element_wise_add(); });
    std::cout << "  Element-wise add:" << std::endl;
    std::cout << "    Mean: " << std::fixed << std::setprecision(3) << t_add << " ms" << std::endl;
    std::cout << std::endl;

    double t_mul = measure_ms([&]() { bench.bench_scalar_mul(); });
    std::cout << "  Scalar multiply:" << std::endl;
    std::cout << "    Mean: " << t_mul << " ms" << std::endl;
    std::cout << std::endl;

    double t_dot = measure_ms([&]() { sink = bench.bench_dot_product(); });
    std::cout << "  Dot product:" << std::endl;
    std::cout << "    Mean: " << t_dot << " ms" << std::endl;
    std::cout << std::endl;

    double t_sum = measure_ms([&]() { sink = bench.bench_vector_sum(); });
    std::cout << "  Vector sum:" << std::endl;
    std::cout << "    Mean: " << t_sum << " ms" << std::endl;
    std::cout << std::endl;

    double t_sqrt = measure_ms([&]() { bench.bench_sqrt(); });
    std::cout << "  Sqrt:" << std::endl;
    std::cout << "    Mean: " << t_sqrt << " ms" << std::endl;
    std::cout << std::endl;
}

int main() {
    std::cout << "=" << std::string(68, ' ') << "=" << std::endl;
    std::cout << "  C++ std::vector Benchmark" << std::endl;
    std::cout << "=" << std::string(68, ' ') << "=" << std::endl;
    std::cout << std::endl;

    std::cout << "Operations: element-wise add, scalar mul, dot product, sum, sqrt" << std::endl;
    std::cout << std::endl;

    std::cout << "--- Small arrays (N=10000) ---" << std::endl;
    run_benchmark<10000>();

    std::cout << "--- Medium arrays (N=100000) ---" << std::endl;
    run_benchmark<100000>();

    std::cout << "--- Large arrays (N=1000000) ---" << std::endl;
    run_benchmark<1000000>();

    std::cout << "=" << std::string(68, ' ') << "=" << std::endl;
    std::cout << "  C++ std::vector Benchmark Complete" << std::endl;
    std::cout << "=" << std::string(68, ' ') << "=" << std::endl;

    return 0;
}