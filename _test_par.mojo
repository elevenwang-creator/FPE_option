from std.algorithm import parallelize

fn main() raises:
    var results = List[Int]()
    for _ in range(8):
        results.append(0)
    
    @parameter
    fn worker(i: Int):
        results[i] = i * i

    parallelize[worker](8)

    var total = 0
    for i in range(8):
        total += results[i]
    print("parallelize sum:", total)
