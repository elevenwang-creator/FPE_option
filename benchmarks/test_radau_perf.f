C Simple Fortran 77 performance test program for RADAU5-like calculations
C Compares basic numerical operations similar to RADAU5

      PROGRAM RADAU5_PERF_TEST
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER I, J, K_ITER, ITERS
      PARAMETER (N=3)
      DOUBLE PRECISION Y(N), Y0(N), M(N,N), KMAT(N,N), E1(N,N)
      DOUBLE PRECISION Z1(N), Z2(N), Z3(N), F(N)
      DOUBLE PRECISION T1, T2, ELAPSED, PER_ITER
      INTEGER ITIME(2)
      INTEGER, EXTERNAL :: ETIME
      DOUBLE PRECISION :: TARRAY(2)

C Initialize system: y' = -KMAT*y, KMAT = diag(1,2,3), M=I
      ITERS = 1000

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
      PRINT *, '  Fortran 77 Performance Test (RADAU5-like operations)'
      PRINT *, '='//REPEAT('=',68)
      PRINT *, ''

C Warmup
      DO K_ITER = 1, 10
          CALL SIMULATE_STEP(N, Y, Y0, M, KMAT, Z1, Z2, Z3, F, E1)
      END DO

C Benchmark
      T1 = ETIME(TARRAY)
      DO K_ITER = 1, ITERS
          CALL SIMULATE_STEP(N, Y, Y0, M, KMAT, Z1, Z2, Z3, F, E1)
      END DO
      T2 = ETIME(TARRAY)
      ELAPSED = T2 - T1
      PER_ITER = ELAPSED / DBLE(ITERS)

      PRINT *, '[1] Small system (n=3):'
      PRINT *, '  Iterations: ', ITERS
      PRINT *, '  Total time: ', ELAPSED, ' s'
      PRINT *, '  Per iteration: ', PER_ITER * 1000.0D0, ' ms'
      PRINT *, '  Throughput: ', DBLE(ITERS) / ELAPSED, ' iters/s'
      PRINT *, ''

      PRINT *, '='//REPEAT('=',68)
      PRINT *, '  Fortran 77 Benchmark complete'
      PRINT *, '='//REPEAT('=',68)

      STOP
      END

C Simulates one step of RADAU5-like calculations
      SUBROUTINE SIMULATE_STEP(N, Y, Y0, M, KMAT, Z1, Z2, Z3, F, E1)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER N
      DOUBLE PRECISION Y(N), Y0(N), M(N,N), KMAT(N,N), E1(N,N)
      DOUBLE PRECISION Z1(N), Z2(N), Z3(N), F(N)
      DOUBLE PRECISION H, U1, DD1, DD2, DD3, TMP

C RADAU5 coefficients (simplified)
      H = 0.1D0
      U1 = 1.0D0
      DD1 = -1.0D0
      DD2 = -1.0D0
      DD3 = -1.0D0

C Compute E1 matrix (E1 = U1*M + h*KMAT)
      DO I = 1, N
          DO J = 1, N
              E1(I,J) = U1 * M(I,J) + H * KMAT(I,J)
          END DO
      END DO

C Newton iteration simulation
      DO NEWT = 1, 7
C Compute function evaluation
          DO I = 1, N
              F(I) = 0.0D0
              DO J = 1, N
                  F(I) = F(I) - KMAT(I,J) * Y(J)
              END DO
          END DO

C Update solution
          DO I = 1, N
              Y(I) = Y(I) + H * 0.1D0 * F(I)
          END DO
      END DO

C Error estimation simulation
      DO I = 1, N
          Z1(I) = DD1 * Y(I) / H
          Z2(I) = DD2 * Y(I) / H
          Z3(I) = DD3 * Y(I) / H
      END DO

      RETURN
      END
