@fieldwise_init
struct ProjectedGradient:
    """Non-negative least squares via projected gradient.
    Solves: min 0.5 ||Ac - b||² s.t. c >= 0.
    """

    var max_iter: Int
    var tol: Float64
    var step_size: Float64

    def solve(self, A: List[List[Float64]], b: List[Float64]) raises -> List[Float64]:
        if len(b) == 0:
            return []
        if len(A) != len(b):
            raise Error("ProjectedGradient: A row count must equal len(b)")
        if len(A[0]) == 0:
            return []

        var n = len(A[0])
        var m = len(b)
        for i in range(m):
            if len(A[i]) != n:
                raise Error("ProjectedGradient: A must be rectangular")

        var c: List[Float64] = []
        for _ in range(n):
            c.append(0.0)

        var step = self.step_size
        if step <= 0.0:
            var max_diag = 0.0
            for j in range(n):
                var diag_j = 0.0
                for i in range(m):
                    diag_j += A[i][j] * A[i][j]
                if diag_j > max_diag:
                    max_diag = diag_j
            step = 1.0 / (max_diag + 1e-10)

        for _ in range(self.max_iter):
            var r: List[Float64] = []
            for i in range(m):
                var ri = -b[i]
                for j in range(n):
                    ri += A[i][j] * c[j]
                r.append(ri)

            var g: List[Float64] = []
            var g_inf = 0.0
            for j in range(n):
                var gj = 0.0
                for i in range(m):
                    gj += A[i][j] * r[i]
                g.append(gj)

                var abs_gj = gj
                if abs_gj < 0.0:
                    abs_gj = -abs_gj
                if abs_gj > g_inf:
                    g_inf = abs_gj

            if g_inf < self.tol:
                break

            for j in range(n):
                var new_cj = c[j] - step * g[j]
                c[j] = new_cj if new_cj > 0.0 else 0.0

        return c^


@fieldwise_init
struct OSQP:
    """ADMM-based QP solver. Delegates to ProjectedGradient for NNLS problems."""

    var max_iter: Int
    var tol: Float64

    def solve_nnls(self, A: List[List[Float64]], b: List[Float64]) raises -> List[Float64]:
        var pg = ProjectedGradient(max_iter=self.max_iter, tol=self.tol, step_size=-1.0)
        return pg.solve(A, b)
