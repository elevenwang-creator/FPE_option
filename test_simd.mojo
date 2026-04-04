from std.sys import simd_width_of
def main():
    comptime width = simd_width_of[DType.float64]()
    var a = SIMD[DType.float64, width](1.0)
    var b = SIMD[DType.float64, width](2.0)
    var c = a >= b
    print(width)
