from std.benchmark import Bench, BenchConfig, BenchId, Bencher

def main() raises:
    var bench = Bench(BenchConfig(max_iters=1000))
    print("Benchmarking GPU batch pricing...")
    print(bench)
