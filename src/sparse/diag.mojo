"""Diagonal matrix with direct CSR construction."""

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


def identity_csr(n: Int) -> CSRMatrix:
    var values: List[Float64] = []
    for i in range(n):
        values.append(1.0)
    return DiagMatrix(values^).to_csr()
