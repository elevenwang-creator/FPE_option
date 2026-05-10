"""Diagonal matrix with direct CSR construction and operator overloads.

DiagMatrix is a lightweight representation storing only the diagonal values.
Operators return DiagMatrix (cheap element-wise on small values list).
Callers call .to_csr() when CSR format is needed.
"""

from sparse.csr import CSRMatrix


struct DiagMatrix(Movable):
    var values: List[Float64]
    var size: Int

    def __init__(out self, var values: List[Float64]):
        self.size = len(values)
        self.values = values^

    def to_csr(self) -> CSRMatrix:
        var n = self.size
        var nnz_count = 0
        for i in range(n):
            if self.values[i] != 0:
                nnz_count += 1

        var result = CSRMatrix(n, n, nnz_count)
        result.indptr[0] = 0
        var dest = 0
        for i in range(n):
            if self.values[i] != 0:
                result.data[dest] = self.values[i]
                result.indices[dest] = i
                dest += 1
            result.indptr[i + 1] = dest

        return result^

    def __add__(self, rhs: Self) -> Self:
        var n = self.size
        var result: List[Float64] = []
        for i in range(n):
            result.append(self.values[i] + rhs.values[i])
        return Self(result^)

    def __matmul__(self, rhs: Self) -> Self:
        var n = self.size
        var result: List[Float64] = []
        for i in range(n):
            result.append(self.values[i] * rhs.values[i])
        return Self(result^)

    def __rmul__(self, alpha: Float64) -> Self:
        var n = self.size
        var result: List[Float64] = []
        for i in range(n):
            result.append(alpha * self.values[i])
        return Self(result^)


def identity_csr(n: Int) -> CSRMatrix:
    var values: List[Float64] = []
    for i in range(n):
        values.append(1.0)
    return DiagMatrix(values^).to_csr()
