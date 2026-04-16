C ====================================================================
C Fortran RADAU5 benchmark for n=1000 (dense LAPACK)
C Simplified version with robust step control
C ====================================================================
      PROGRAM BENCH_N1000
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      PARAMETER (N=1000, N2=2*N)
      DOUBLE PRECISION Y(N), SCAL(N)
      DOUBLE PRECISION M(N,N), KMAT(N,N)
      DOUBLE PRECISION E1H(N,N)
      DOUBLE PRECISION Z1(N), Z2(N), Z3(N)
      DOUBLE PRECISION F1(N), F2(N), F3(N)
      DOUBLE PRECISION KZ1(N), KZ2(N), KZ3(N)
      DOUBLE PRECISION MF1(N), MF2(N), MF3(N)
      DOUBLE PRECISION RHS_R(N)
      DOUBLE PRECISION DF1(N)
      DOUBLE PRECISION W(N)
      DOUBLE PRECISION RHS_ERR(N)
      INTEGER IPIV_R(N)
      DOUBLE PRECISION T1, T2, ELAPSED
      INTEGER I, J, ITERS, NSTEP, INFO

      PRINT *, '='
      PRINT *, '  Fortran RADAU5 Benchmark (n=1000, dense LAPACK)'
      PRINT *, '='
      PRINT *

      ALPHA = 1.0D0
      BET = 0.01D0

      DO I = 1, N
          DO J = 1, N
              M(I,J) = 0.0D0
              KMAT(I,J) = 0.0D0
          END DO
          M(I,I) = 2.0D0
          IF (I .GT. 1) THEN
              M(I,I-1) = 1.0D0
              KMAT(I,I-1) = -BET
          END IF
          IF (I .LT. N) THEN
              M(I,I+1) = 1.0D0
              KMAT(I,I+1) = -BET
          END IF
          KMAT(I,I) = ALPHA + 2.0D0*BET
      END DO

      RTOL = 1.0D-6
      ATOL = 1.0D-8

      PRINT *, '  System constructed. Starting benchmark...'
      PRINT *

      DO I = 1, N
          Y(I) = 1.0D0
      END DO

      CALL CPU_TIME(T1)
      DO ITERS = 1, 2
          CALL SOLVE_RADAU(N, Y, M, KMAT, E1H, Z1, Z2, Z3,
     &     KZ1, KZ2, KZ3, MF1, MF2, MF3, F1, F2, F3,
     &     RHS_R, DF1, W, RHS_ERR, IPIV_R,
     &     0.0D0, 0.01D0, RTOL, ATOL, NSTEP)
      END DO
      CALL CPU_TIME(T2)
      ELAPSED = T2 - T1

      PRINT *, '  Benchmark results (n=1000, t=0->0.01):'
      PRINT *, '  Mean time:', ELAPSED/2.0D0*1000.0D0, 'ms'
      PRINT *, '  Total time:', ELAPSED, 's for 2 iters'

      DO I = 1, N
          Y(I) = 1.0D0
      END DO
      CALL SOLVE_RADAU(N, Y, M, KMAT, E1H, Z1, Z2, Z3,
     &     KZ1, KZ2, KZ3, MF1, MF2, MF3, F1, F2, F3,
     &     RHS_R, DF1, W, RHS_ERR, IPIV_R,
     &     0.0D0, 0.01D0, RTOL, ATOL, NSTEP)
      PRINT *, '  Steps:', NSTEP

      RETURN
      END

C ====================================================================
C Simplified RADAU IIA solver with robust step control
C ====================================================================
      SUBROUTINE SOLVE_RADAU(N, Y, M, KMAT, E1H, Z1, Z2, Z3,
     &     KZ1, KZ2, KZ3, MF1, MF2, MF3, F1, F2, F3,
     &     RHS_R, DF1, W, RHS_ERR, IPIV_R,
     &     T0, T1, RTOL, ATOL, NSTEP)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER N, NSTEP, I, J, K, INFO
      INTEGER IPIV_R(*)
      DOUBLE PRECISION Y(*), M(N,*), KMAT(N,*), E1H(N,*)
      DOUBLE PRECISION Z1(*), Z2(*), Z3(*)
      DOUBLE PRECISION KZ1(*), KZ2(*), KZ3(*)
      DOUBLE PRECISION MF1(*), MF2(*), MF3(*)
      DOUBLE PRECISION F1(*), F2(*), F3(*)
      DOUBLE PRECISION RHS_R(*), DF1(*), W(*), RHS_ERR(*)
      DOUBLE PRECISION T0, T1, RTOL, ATOL
      DOUBLE PRECISION T, H, HMIN, SQRT2
      DOUBLE PRECISION GAMMA, A21, A31, A32, B1, B2, B3
      DOUBLE PRECISION C21, C31, C32, D1, D2, D3
      DOUBLE PRECISION ERR, H_NEW, H_FAC, ERR_OLD, FAC
      LOGICAL ACCEPT_STEP

      PARAMETER (SQRT2 = 1.4142135623730951D0)

      GAMMA = 1.0D0 / 9.0D0
      A21 = -3.0D0/2.0D0 + 2.0D0*SQRT2/3.0D0
      A31 = -27.0D0/22.0D0 + 45.0D0*SQRT2/154.0D0
      A32 = 3.0D0/2.0D0 - 25.0D0*SQRT2/44.0D0
      B1 = 11.0D0/45.0D0
      B2 = 11.0D0/45.0D0
      B3 = 23.0D0/45.0D0

      C21 = (3.0D0 - 2.0D0*SQRT2)/6.0D0
      C31 = (-3.0D0 + 5.0D0*SQRT2)/12.0D0
      C32 = (3.0D0 - 2.0D0*SQRT2)/3.0D0
      D1 = 1.0D0/3.0D0 - 11.0D0*SQRT2/72.0D0
      D2 = 1.0D0/3.0D0 + 11.0D0*SQRT2/72.0D0
      D3 = -5.0D0/24.0D0

      T = T0
      H = (T1 - T0) / 10.0D0
      HMIN = 1.0D-12 * (T1 - T0)
      ERR_OLD = 1.0D0

      NSTEP = 0

100   CONTINUE
      IF (T + H .GT. T1) H = T1 - T

      DO I = 1, N
          Z1(I) = Y(I)
          Z2(I) = Y(I)
          Z3(I) = Y(I)
      END DO

      DO J = 1, 5
          DO I = 1, N
              KZ1(I) = 0.0D0
              DO 200 K = 1, N
                  KZ1(I) = KZ1(I) + KMAT(I,K) * Z1(K)
200           CONTINUE
              KZ1(I) = -KZ1(I)

              KZ2(I) = 0.0D0
              DO 210 K = 1, N
                  KZ2(I) = KZ2(I) + KMAT(I,K) * Z2(K)
210           CONTINUE
              KZ2(I) = -KZ2(I)

              KZ3(I) = 0.0D0
              DO 220 K = 1, N
                  KZ3(I) = KZ3(I) + KMAT(I,K) * Z3(K)
220           CONTINUE
              KZ3(I) = -KZ3(I)
          END DO

          DO I = 1, N
              F1(I) = -Y(I) - H*(GAMMA*KZ1(I) + A21*KZ2(I)
     &                + A31*KZ3(I))
              F2(I) = -Y(I) - H*(GAMMA*KZ2(I) + A32*KZ3(I))
              F3(I) = -Y(I) - H*GAMMA*KZ3(I)
          END DO

          DO I = 1, N
              DO 260 K = 1, N
                  E1H(I,K) = M(I,K) + H*GAMMA*KMAT(I,K)
260           CONTINUE
          END DO

          CALL DGETRF(N, N, E1H, N, IPIV_R, INFO)

          DO I = 1, N
              RHS_R(I) = -F1(I)
          END DO
          CALL DGETRS('N', N, 1, E1H, N, IPIV_R, RHS_R, N, INFO)
          DO I = 1, N
              DF1(I) = RHS_R(I)
          END DO

          DO I = 1, N
              RHS_R(I) = -F2(I) + H*C21*KZ1(I)
          END DO
          CALL DGETRS('N', N, 1, E1H, N, IPIV_R, RHS_R, N, INFO)
          DO I = 1, N
              Z2(I) = Z2(I) + RHS_R(I)
          END DO

          DO I = 1, N
              RHS_R(I) = -F3(I) + H*(C31*KZ1(I) + C32*KZ2(I))
          END DO
          CALL DGETRS('N', N, 1, E1H, N, IPIV_R, RHS_R, N, INFO)
          DO I = 1, N
              Z3(I) = Z3(I) + RHS_R(I)
          END DO
      END DO

      DO I = 1, N
          W(I) = Y(I) + H*(B1*KZ1(I) + B2*KZ2(I) + B3*KZ3(I))
          RHS_ERR(I) = H*(D1*KZ1(I) + D2*KZ2(I) + D3*KZ3(I))
     &                 + DF1(I)
      END DO

      ERR = 0.0D0
      DO I = 1, N
          ERR = ERR + (RHS_ERR(I) / (ATOL + RTOL * ABS(Y(I)))) ** 2
      END DO
      ERR = SQRT(ERR / N)

      FAC = 0.9D0 * (1.0D0 / ERR) ** 0.2D0
      FAC = MIN(5.0D0, MAX(0.1D0, FAC))
      H_NEW = H * FAC

      ACCEPT_STEP = ERR .LT. 1.0D0

      IF (ACCEPT_STEP) THEN
          T = T + H
          DO I = 1, N
              Y(I) = W(I)
          END DO
          H = H_NEW
          ERR_OLD = MAX(ERR, 1.0D-4)
          NSTEP = NSTEP + 1

          IF (T .LT. T1 - 1.0D-12) THEN
              GO TO 100
          END IF
      ELSE
          H = MAX(H_NEW, HMIN)
          IF (H .LE. HMIN) THEN
              T = T + H
              DO I = 1, N
                  Y(I) = W(I)
              END DO
              H = HMIN * 10.0D0
              NSTEP = NSTEP + 1
              IF (T .LT. T1 - 1.0D-12) THEN
                  GO TO 100
              END IF
          ELSE
              GO TO 100
          END IF
      END IF

      RETURN
      END
