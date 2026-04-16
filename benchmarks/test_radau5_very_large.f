C Fortran 77 very large matrix benchmark (n=1000)
C Uses a tridiagonal K matrix for large-scale performance testing
C Compiles with macOS Accelerate framework

      PROGRAM RADAU5_VERY_LARGE_MATRIX
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER I, J, K_STEP, NEWT, ITERS
      PARAMETER (N=1000, NS=3, MAX_STEPS=100)
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
      PARAMETER (U1=3.6378165692476072D0,
     &           DD1=(-13.0D0 - 7.0D0*2.449489742783178D0)/3.0D0,
     &           DD2=(-13.0D0 + 7.0D0*2.449489742783178D0)/3.0D0,
     &           DD3=-1.0D0/3.0D0)

      ITERS = 3
      TSTART = 0.0D0
      TEND = 0.1D0
      H0 = 0.001D0
      RTOL = 1.0D-3
      ATOL = 1.0D-6

      PRINT *, '='//REPEAT('=',68)
      PRINT *, '  Fortran 77 Very Large Matrix Benchmark (n=1000)'
      PRINT *, '  (Tridiagonal K matrix, complete integration)'
      PRINT *, '='//REPEAT('=',68)
      PRINT *, ''

C Initialize system: tridiagonal K matrix
      DO I = 1, N
          Y0(I) = 1.0D0
          SCAL(I) = 1.0D0
          DO J = 1, N
              M(I,J) = 0.0D0
              KMAT(I,J) = 0.0D0
          END DO
          M(I,I) = 1.0D0
          KMAT(I,I) = 2.0D0
          IF (I .GT. 1) KMAT(I,I-1) = -1.0D0
          IF (I .LT. N) KMAT(I,I+1) = -1.0D0
      END DO

C Warmup (1 iteration)
      DO K_ITER = 1, 1
          DO I = 1, N
              Y(I) = 1.0D0
          END DO
          CALL COMPLETE_INTEGRATION(N, NS, MAX_STEPS, Y, Y0, YT,
     &         FY, M, KMAT, E1_H, Z1, Z2, Z3, ERR, DZ, SCAL,
     &         TSTART, TEND, H0, RTOL, ATOL,
     &         IPIV, INFO, DGETRF, DGETRS, U1, DD1, DD2, DD3)
      END DO

C Benchmark
      CALL CPU_TIME(T1)
      DO K_ITER = 1, ITERS
          DO I = 1, N
              Y(I) = 1.0D0
          END DO
          CALL COMPLETE_INTEGRATION(N, NS, MAX_STEPS, Y, Y0, YT,
     &         FY, M, KMAT, E1_H, Z1, Z2, Z3, ERR, DZ, SCAL,
     &         TSTART, TEND, H0, RTOL, ATOL,
     &         IPIV, INFO, DGETRF, DGETRS, U1, DD1, DD2, DD3)
      END DO
      CALL CPU_TIME(T2)
      ELAPSED = T2 - T1
      PER_ITER = ELAPSED / DBLE(ITERS)

      PRINT *, '[1] Very large system (n=1000, t=0→0.1):'
      PRINT *, '  Iterations: ', ITERS
      PRINT *, '  Total time: ', ELAPSED, ' s'
      PRINT *, '  Per iteration: ', PER_ITER * 1000.0D0, ' ms'
      PRINT *, '  Throughput: ', DBLE(ITERS) / ELAPSED, ' iters/s'
      PRINT *, ''

      PRINT *, '='//REPEAT('=',68)
      PRINT *, '  Fortran Very Large Matrix Benchmark complete'
      PRINT *, '='//REPEAT('=',68)

      STOP
      END

C --------------------------------------------------------------------
C Complete integration from TSTART to TEND (same as Mojo's solve())
C --------------------------------------------------------------------
      SUBROUTINE COMPLETE_INTEGRATION(N, NS, MAX_STEPS, Y, Y0, YT,
     &         FY, M, KMAT, E1_H, Z1, Z2, Z3, ERR, DZ, SCAL,
     &         TSTART, TEND, H0, RTOL, ATOL,
     &         IPIV, INFO, DGETRF, DGETRS, U1, DD1, DD2, DD3)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER N, NS, MAX_STEPS, IPIV(N), INFO
      DOUBLE PRECISION Y(N), Y0(N), YT(N,NS), FY(N,NS)
      DOUBLE PRECISION M(N,N), KMAT(N,N), E1_H(N,N)
      DOUBLE PRECISION Z1(N), Z2(N), Z3(N), ERR(N), DZ(N), SCAL(N)
      DOUBLE PRECISION TSTART, TEND, H0, RTOL, ATOL
      EXTERNAL DGETRF, DGETRS
      DOUBLE PRECISION U1, DD1, DD2, DD3
      DOUBLE PRECISION T, H, H_OLD, ERR_NORM, ERR_NORM_OLD
      INTEGER NRHS
      CHARACTER TRANS

      T = TSTART
      H = H0
      H_OLD = H
      ERR_NORM_OLD = 1.0D-4

C Copy initial Y
      DO I = 1, N
          Y0(I) = Y(I)
      END DO

C Main integration loop (100 steps from 0→0.1, h=0.001)
      DO K_STEP = 1, 100
          IF (T + 1.01D0*H .GT. TEND) H = TEND - T

C Build E1_h and factorize (LU)
          DO I = 1, N
              DO J = 1, N
                  E1_H(I,J) = U1*M(I,J) + H*KMAT(I,J)
              END DO
          END DO
          CALL DGETRF(N, N, E1_H, N, IPIV, INFO)

C Initial guess
          DO I = 1, N
              YT(I,1) = Y(I)
              YT(I,2) = Y(I)
              YT(I,3) = Y(I)
          END DO

C Simplified Newton iterations (up to 7)
          DO NEWT = 1, 7
C Compute F(Y^i)
              DO I = 1, N
                  FY(I,1) = 0.0D0
                  FY(I,2) = 0.0D0
                  FY(I,3) = 0.0D0
                  DO J = 1, N
                      FY(I,1) = FY(I,1) - KMAT(I,J)*YT(J,1)
                      FY(I,2) = FY(I,2) - KMAT(I,J)*YT(J,2)
                      FY(I,3) = FY(I,3) - KMAT(I,J)*YT(J,3)
                  END DO
              END DO

C Compute RHS and solve
              DO I = 1, N
                  DZ(I) = M(I,I)*Y0(I) + H*FY(I,1)
              END DO
              TRANS = 'N'
              NRHS = 1
              CALL DGETRS(TRANS, N, NRHS, E1_H, N, IPIV, DZ, N, INFO)

C Update solution
              DO I = 1, N
                  YT(I,1) = DZ(I)
                  YT(I,2) = DZ(I)
                  YT(I,3) = DZ(I)
              END DO
          END DO

C Update state
          DO I = 1, N
              Y(I) = YT(I,1)
              Y0(I) = Y(I)
          END DO

C Error estimation
          DO I = 1, N
              Z1(I) = YT(I,1)/H
              Z2(I) = YT(I,2)/H
              Z3(I) = YT(I,3)/H
              ERR(I) = -(DD1*Z1(I) + DD2*Z2(I) + DD3*Z3(I))
          END DO

C Error norm
          ERR_NORM = 0.0D0
          DO I = 1, N
              S = ABS(ERR(I)) / (1.0D0 + MAX(ABS(Y0(I)), ABS(Y(I))))
              ERR_NORM = ERR_NORM + S*S
          END DO
          ERR_NORM = SQRT(ERR_NORM / DBLE(N))

          T = T + H
      END DO

      RETURN
      END
