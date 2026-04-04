"""Benchmark NAIS-Net inference performance."""

from engines.nais.nais_net import NaisNet
from std.python import Python


def bench_nais_forward() raises:
    var net = NaisNet(in_dim=3, hidden=12, phi_dim=2)
    var t = 0.1
    var x: List[Float64] = [100.0, 0.04, 0.0]

    var time_mod = Python.import_module("time")
    var start = time_mod.perf_counter()
    var iterations = 10000
    for _ in range(iterations):
        var _ = net.forward(t, x)
    var end = time_mod.perf_counter()
    var elapsed = Float64(py=end) - Float64(py=start)
    var per_call = elapsed / Float64(iterations) * 1e6

    print("NAIS-Net Forward Pass (in_dim=3, hidden=12, phi_dim=2)")
    print("  Iterations:", iterations)
    print("  Total time:", elapsed, "s")
    print("  Per call:", per_call, "μs")


def main() raises:
    print("=== NAIS Inference Benchmark ===")
    bench_nais_forward()
