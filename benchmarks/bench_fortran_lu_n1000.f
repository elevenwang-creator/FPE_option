C ====================================================================
C Simple benchmark: Compare dense LAPACK vs sparse LU for n=1000
C Only measures LU decomposition time (not full ODE solve)
C ====================================================================
      PROGRAM BENCH_LU_N1000
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (N=1000)
      DOUBLE PRECISION A(N,N)
      INTEGER IPIV(N)
      DOUBLE PRECISION T1, T2, ELAPSED
      INTEGER I, J, ITERS

      PRINT *, '='
      PRINT *, '  Dense LAPACK LU Decomposition (n=1000)'
      PRINT *, '='
      PRINT *

      DO I = 1, N
          DO J = 1, N
              A(I,J) = 0.0D0
          END DO
          A(I,I) = 2.0D0
          IF (I .GT. 1) A(I,I-1) = 1.0D0
          IF (I .LT. N) A(I,I+1) = 1.0D0
      END DO

      PRINT *, '  Matrix constructed. Starting benchmark...'
      PRINT *

      CALL CPU_TIME(T1)
      DO ITERS = 1, 10
          DO I = 1, N
              DO J = 1, N
                  A(I,J) = A(I,J) + 0.001D0 * (I + J)
              END DO
          END DO
          CALL DGETRF(N, N, A, N, IPIV, INFO)
      END DO
      CALL CPU_TIME(T2)
      ELAPSED = T2 - T1

      PRINT *, '  Mean time per LU:', ELAPSED/10.0D0*1000.0D0, 'ms'
      PRINT *, '  Total time:', ELAPSED, 's for 10 iters'

      RETURN
      END
