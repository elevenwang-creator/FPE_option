from std.python import Python
from std.sys import has_accelerator
from engines.fpe.gpu.executor import GPUFullChainExecutor


def bench_batch_size[batch_size: Int](iterations: Int = 5) raises:
    print("Testing Batch Size:", batch_size, "- Iterations:", iterations)
    try:
        var executor = GPUFullChainExecutor[batch_size](n_s=8, n_v=8)
        # Warmup
        executor.execute_batch_pricing()

        var time_mod = Python.import_module("time")
        var start = time_mod.perf_counter()

        for _ in range(iterations):
            executor.execute_batch_pricing()

        var end = time_mod.perf_counter()
        var elapsed = Float64(py=end) - Float64(py=start)

        print("  Total CPU-blocking time:", elapsed, "s")
        print("  Time per batch iteration:", elapsed / Float64(iterations), "s")
        print(
            "  Throughput:",
            Float64(batch_size * iterations) / elapsed,
            "options/sec",
        )
        print("--------------------------------------------------")
    except e:
        print("  [ERROR] Kernel execution failed:", e)


def main() raises:
    if not has_accelerator():
        print("Skipping GPU benchmark: no accelerator found.")
        return

    print("=== Multi-Batch GPU Allocation Benchmark (Metal) ===")

    # Measure typical latency sizes matching Grid/Block distribution
    bench_batch_size[256](10)  # Warmup + small scale
    bench_batch_size[1000](10)  # Medium scale GPU distribution
    bench_batch_size[5000](5)  # Large scale Block distribution
    bench_batch_size[10000](5)  # Edge scale Grid mapping

    print("Benchmark complete!")
