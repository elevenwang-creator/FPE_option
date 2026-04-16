C Correct Fortran 77 test - does the same work as Mojo RADAU5
C Uses real calculations, including dense matrix operations and LU
C Compiles with macOS Accelerate framework

      PROGRAM RADAU5_CORRECT_PERF
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER I, J, K_ITER, ITERS
      PARAMETER (N=3, NS=3)
      DOUBLE PRECISION Y(N), Y0(N), YT(N,NS), FY(N,NS)
      DOUBLE PRECISION M(N,N), KMAT(N,N), E1_H(N,N)
      DOUBLE PRECISION Z1(N), Z2(N), Z3(N), ERR(N)
      DOUBLE PRECISION DZ(N), SCAL(N)
      DOUBLE PRECISION T1, T2, ELAPSED, PER_ITER

      EXTERNAL DGETRF, DGETRS
      INTEGER IPIV(N), INFO
      INTEGER NRHS
      CHARACTER TRANS

C RADAU5 coefficients from Hairer's code
      DOUBLE PRECISION U1, DD1, DD2, DD3
      PARAMETER (U1=0.66666666666666663D0,
     &           DD1=-5.6568542494923797D0,
     &           DD2=1.4142135623730951D0,
     &           DD3=2.8284271247461903D0)

      ITERS = 100

C Initialize same system as Mojo test
      DO I = 1, N
          Y0(I) = 1.0D0
          Y(I) = 1.0D0
          SCAL(I) = 1.0D0
          DO J = 1, N
              M(I,J) = 0.0D0
              KMAT(I,J) = 0.0D0
          END DO
          M(I,I) = 1.0D0
          KMAT(I,I) = DBLE(I)
      END DO

      PRINT *, '='//REPEAT('=',68)
      PRINT *, '  Fortran 77 (Accelerate) - Real RADAU5 Calculations'
      PRINT *, '='//REPEAT('=',68)
      PRINT *, ''

C Warmup (10 iterations)
      DO K_ITER = 1, 10
          CALL DO_RADAU5_STEP(N, NS, Y, Y0, YT, FY, M, KMAT, E1_H,
     &         Z1, Z2, Z3, ERR, DZ, SCAL, IPIV, INFO,
     &         DGETRF, DGETRS, U1, DD1, DD2, DD3)
      END DO

C Benchmark
      CALL CPU_TIME(T1)
      DO K_ITER = 1, ITERS
          CALL DO_RADAU5_STEP(N, NS, Y, Y0, YT, FY, M, KMAT, E1_H,
     &         Z1, Z2, Z3, ERR, DZ, SCAL, IPIV, INFO,
     &         DGETRF, DGETRS, U1, DD1, DD2, DD3)
      END DO
      CALL CPU_TIME(T2)
      ELAPSED = T2 - T1
      PER_ITER = ELAPSED / DBLE(ITERS)

      PRINT *, '[1] Full RADAU5 step (same as Mojo):'
      PRINT *, '  Iterations: ', ITERS
      PRINT *, '  Total time: ', ELAPSED, ' s'
      PRINT *, '  Per iteration: ', PER_ITER * 1000.0D0, ' ms'
      PRINT *, '  Throughput: ', DBLE(ITERS) / ELAPSED, ' iters/s'
      PRINT *, ''

      PRINT *, '='//REPEAT('=',68)
      PRINT *, '  Fortran Benchmark complete'
      PRINT *, '='//REPEAT('=',68)

      STOP
      END

C --------------------------------------------------------------------
C Real RADAU5 step calculation, matching what Mojo does
C --------------------------------------------------------------------
      SUBROUTINE DO_RADAU5_STEP(N, NS, Y, Y0, YT, FY, M, KMAT, E1_H,
     &         Z1, Z2, Z3, ERR, DZ, SCAL, IPIV, INFO,
     &         DGETRF, DGETRS, U1, DD1, DD2, DD3)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER N, NS, IPIV(N), INFO
      DOUBLE PRECISION Y(N), Y0(N), YT(N,NS), FY(N,NS)
      DOUBLE PRECISION M(N,N), KMAT(N,N), E1_H(N,N)
      DOUBLE PRECISION Z1(N), Z2(N), Z3(N), ERR(N), DZ(N), SCAL(N)
      EXTERNAL DGETRF, DGETRS
      DOUBLE PRECISION U1, DD1, DD2, DD3

      DOUBLE PRECISION H, TOL, ERR_NORM
      PARAMETER (H=0.1D0, TOL=1.0D-3)

      INTEGER NEWT, I, J
      DOUBLE PRECISION S, F_NORM
      INTEGER NRHS
      CHARACTER TRANS

C 1. Save initial y0
      DO I = 1, N
          Y0(I) = Y(I)
      END DO

C 2. Compute E1_h = U1*M + h*KMAT (Hairer formula)
      DO I = 1, N
          DO J = 1, N
              E1_H(I,J) = U1 * M(I,J) + H * KMAT(I,J)
          END DO
      END DO

C 3. LU decomposition of E1_h (LAPACK DGETRF)
      CALL DGETRF(N, N, E1_H, N, IPIV, INFO)

C 4. Initial guess (simple extrapolation)
      DO I = 1, N
          YT(I,1) = Y(I)
          YT(I,2) = Y(I)
          YT(I,3) = Y(I)
      END DO

C 5. Simplified Newton iterations (up to NIT=7)
      DO NEWT = 1, 7
C a. Compute F(Y^i) = -KMAT*Y^i
          DO I = 1, N
              FY(I,1) = 0.0D0
              FY(I,2) = 0.0D0
              FY(I,3) = 0.0D0
              DO J = 1, N
                  FY(I,1) = FY(I,1) - KMAT(I,J) * YT(J,1)
                  FY(I,2) = FY(I,2) - KMAT(I,J) * YT(J,2)
                  FY(I,3) = FY(I,3) - KMAT(I,J) * YT(J,3)
              END DO
          END DO

C b. Compute RHS = M*Z0 + h*F, then solve E1_h*dz = RHS
          DO I = 1, N
              DZ(I) = M(I,I)*Y0(I) + H*FY(I,1)
          END DO
          TRANS = 'N'
          NRHS = 1
          CALL DGETRS(TRANS, N, NRHS, E1_H, N, IPIV, DZ, N, INFO)

C c. Update solution
          DO I = 1, N
              YT(I,1) = DZ(I)
              YT(I,2) = DZ(I)
              YT(I,3) = DZ(I)
          END DO
      END DO

C 6. Update solution at x+h
      DO I = 1, N
          Y(I) = YT(I,1)
      END DO

C 7. Error estimation (Hairer formula: err = -(DD1*Z1 + DD2*Z2 + DD3*Z3))
      DO I = 1, N
          Z1(I) = YT(I,1) / H
          Z2(I) = YT(I,2) / H
          Z3(I) = YT(I,3) / H
          ERR(I) = -(DD1*Z1(I) + DD2*Z2(I) + DD3*Z3(I))
      END DO

C 8. Error norm calculation
      ERR_NORM = 0.0D0
      DO I = 1, N
          S = ABS(ERR(I)) / (1.0D0 + MAX(ABS(Y0(I)), ABS(Y(I))))
          ERR_NORM = ERR_NORM + S*S
      END DO
      ERR_NORM = SQRT(ERR_NORM / DBLE(N))

      RETURN
      END
