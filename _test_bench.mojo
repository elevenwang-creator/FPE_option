from std.benchmark import run as bench_run

fn bench_fn():
    var x = 0
    for i in range(1000):
        x += i
    _ = x

fn main():
    var report = bench_run[bench_fn]()
    print(report)
