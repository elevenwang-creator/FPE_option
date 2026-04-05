from sparse.csr import CSRMatrix


struct DiagMatrix[dtype: DType](Copyable, Movable):
    var diag: List[Scalar[Self.dtype]]
    var size: Int

    def __init__(out self, size: Int):
        self.size = size
        self.diag = []
        for _ in range(size):
            self.diag.append(0)

    def __init__(out self, var diag: List[Scalar[Self.dtype]]):
        self.size = len(diag)
        self.diag = diag^

    def diag_vec_mul(self, x: List[Scalar[Self.dtype]]) -> List[Scalar[Self.dtype]]:
        from std.sys import simd_width_of
        comptime width = simd_width_of[DType.float64]()
        var limit = self.size
        if len(x) < limit:
            limit = len(x)
        var y: List[Scalar[Self.dtype]] = []
        for _ in range(self.size):
            y.append(0)

        var i = 0
        while i + width <= limit:
            var vals = SIMD[Self.dtype, width]()
            var x_vals = SIMD[Self.dtype, width]()
            for k in range(width):
                vals[k] = self.diag[i + k]
                x_vals[k] = x[i + k]
            var res = vals * x_vals
            for k in range(width):
                y[i + k] = res[k]
            i += width

        while i < limit:
            y[i] = self.diag[i] * x[i]
            i += 1
        return y^

    def to_csr(self) -> CSRMatrix[Self.dtype]:
        var out = CSRMatrix[Self.dtype](self.size, self.size)
        out.indptr[0] = 0
        for i in range(self.size):
            var v = self.diag[i]
            if v != 0:
                out.indices.append(i)
                out.data.append(v)
            out.indptr[i + 1] = len(out.data)
        return out^

    def inverse(self) -> Self:
        """Invert diagonal: D⁻¹[i] = 1 / D[i]."""
        var out = Self(self.size)
        for i in range(self.size):
            if self.diag[i] != 0:
                out.diag[i] = 1.0 / self.diag[i]
        return out^

