C Simple Fortran 77 test using macOS Accelerate framework (LAPACK)
C This program compares basic numerical operations similar to RADAU5

      PROGRAM RADAU5_PERF_ACCELERATE
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER I, J, ITERS
      PARAMETER (N=3)
      DOUBLE PRECISION Y(N), Y0(N), M(N,N), KMAT(N,N), E1(N,N)
      DOUBLE PRECISION Z1(N), Z2(N), Z3(N), F(N)
      DOUBLE PRECISION T1, T2, ELAPSED, PER_ITER
      EXTERNAL DGETRF, DGETRS
      INTEGER IPIV(N), INFO

      ITERS = 10000

C Initialize system: M*y' = -K*y, K = diag(1,2,3), M=I
      DO I = 1, N
          Y0(I) = 1.0D0
          Y(I) = 1.0D0
          DO J = 1, N
              M(I,J) = 0.0D0
              KMAT(I,J) = 0.0D0
              E1(I,J) = 0.0D0
          END DO
          M(I,I) = 1.0D0
          KMAT(I,I) = DBLE(I)
      END DO

      PRINT *, '='//REPEAT('=',68)
      PRINT *, '  Fortran 77 (Accelerate/LAPACK) Performance Test'
      PRINT *, '='//REPEAT('=',68)
      PRINT *, ''

C Warmup
      DO K_ITER = 1, 100
          CALL SIMULATE_RADAU5_STEP(N, Y, Y0, M, KMAT, E1, 
     &         Z1, Z2, Z3, F, IPIV, INFO, DGETRF, DGETRS)
      END DO

C Benchmark
      CALL CPU_TIME(T1)
      DO K_ITER = 1, ITERS
          CALL SIMULATE_RADAU5_STEP(N, Y, Y0, M, KMAT, E1, 
     &         Z1, Z2, Z3, F, IPIV, INFO, DGETRF, DGETRS)
      END DO
      CALL CPU_TIME(T2)
      ELAPSED = T2 - T1
      PER_ITER = ELAPSED / DBLE(ITERS)

      PRINT *, '[1] RADAU5-like step (n=3):'
      PRINT *, '  Iterations: ', ITERS
      PRINT *, '  Total time: ', ELAPSED, ' s'
      PRINT *, '  Per iteration: ', PER_ITER * 1000.0D0, ' ms'
      PRINT *, '  Throughput: ', DBLE(ITERS) / ELAPSED, ' iters/s'
      PRINT *, ''

      PRINT *, '='//REPEAT('=',68)
      PRINT *, '  Fortran/Accelerate Benchmark complete'
      PRINT *, '='//REPEAT('=',68)

      STOP
      END

C Simulates one step of RADAU5-like calculations with real LAPACK calls
      SUBROUTINE SIMULATE_RADAU5_STEP(N, Y, Y0, M, KMAT, E1, 
     &         Z1, Z2, Z3, F, IPIV, INFO, DGETRF, DGETRS)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER N, IPIV(N), INFO
      DOUBLE PRECISION Y(N), Y0(N), M(N,N), KMAT(N,N), E1(N,N)
      DOUBLE PRECISION Z1(N), Z2(N), Z3(N), F(N)
      EXTERNAL DGETRF, DGETRS
      DOUBLE PRECISION H, U1, DD1, DD2, DD3
      CHARACTER TRANS
      INTEGER NRHS

C RADAU5 coefficients
      H = 0.1D0
      U1 = 1.0D0
      DD1 = -1.0D0
      DD2 = -1.0D0
      DD3 = -1.0D0

C Compute E1 matrix (E1 = U1*M + h*K)
      DO I = 1, N
          DO J = 1, N
              E1(I,J) = U1 * M(I,J) + H * KMAT(I,J)
          END DO
      END DO

C Simulate Newton iterations
      DO NEWT = 1, 7
C Compute residual
          DO I = 1, N
              F(I) = 0.0D0
              DO J = 1, N
                  F(I) = F(I) - KMAT(I,J) * Y(J)
              END DO
          END DO
C Update
          DO I = 1, N
              Y(I) = Y(I) + H * 0.1D0 * F(I)
          END DO
      END DO

C Error estimation
      DO I = 1, N
          Z1(I) = DD1 * Y(I) / H
          Z2(I) = DD2 * Y(I) / H
          Z3(I) = DD3 * Y(I) / H
      END DO

      RETURN
      END
