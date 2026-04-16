C Fortran 77 self-contained RADAU IIA (order 5, s=3) benchmark
C Same algorithm as Mojo radau.mojo for fair comparison
C Uses LAPACK DGETRF/DGETRS for LU factorization
C
C Compile:
C   gfortran -O2 -o bench_fortran_radau5 bench_fortran_radau5.f
C           -framework Accelerate

      PROGRAM BENCH_RADAU5
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)

      PRINT *, '======================================'
      PRINT *, '  Fortran 77 RADAU5 Benchmark'
      PRINT *, '======================================'
      PRINT *, ''

      PRINT *, '[1] Small Diagonal System (n=3, t=0->5)'
      CALL BENCH_DIAG3()
      PRINT *, ''

      PRINT *, '[2] Medium Tridiagonal System (n=10, t=0->1)'
      CALL BENCH_TRIDIAG10()
      PRINT *, ''

      PRINT *, '[3] Large Tridiagonal System (n=100, t=0->1)'
      CALL BENCH_TRIDIAG100()
      PRINT *, ''

      PRINT *, '======================================'
      PRINT *, '  Fortran Benchmark complete'
      PRINT *, '======================================'

      STOP
      END


C ====================================================================
C Small diagonal system (n=3)
C ====================================================================
      SUBROUTINE BENCH_DIAG3()
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER N, N2
      PARAMETER (N=3, N2=2*N)
      DOUBLE PRECISION Y(N), SCAL(N)
      DOUBLE PRECISION M(N,N), KMAT(N,N)
      DOUBLE PRECISION E1H(N,N), E2H(N2,N2)
      DOUBLE PRECISION Z1(N), Z2(N), Z3(N)
      DOUBLE PRECISION F1(N), F2(N), F3(N)
      DOUBLE PRECISION KZ1(N), KZ2(N), KZ3(N)
      DOUBLE PRECISION MF1(N), MF2(N), MF3(N)
      DOUBLE PRECISION RHS_R(N), RHS_CX(N2)
      DOUBLE PRECISION DF1(N), DF23(N2)
      DOUBLE PRECISION W(N), CONT(N), MCONT(N)
      DOUBLE PRECISION RHS_ERR(N), ERR_VEC(N)
      INTEGER IPIV_R(N), IPIV_CX(N2)
      DOUBLE PRECISION T1, T2, ELAPSED
      INTEGER I, J, ITERS, NSTEP

      DO I = 1, N
          DO J = 1, N
              M(I,J) = 0.0D0
              KMAT(I,J) = 0.0D0
          END DO
          M(I,I) = 1.0D0
      END DO
      KMAT(1,1) = 0.1D0
      KMAT(2,2) = 0.5D0
      KMAT(3,3) = 2.0D0

      RTOL = 1.0D-6
      ATOL = 1.0D-8

      DO ITERS = 1, 3
          DO I = 1, N
              Y(I) = 1.0D0
          END DO
          CALL SOLVE_RADAU(N, N2, Y, M, KMAT, N, N2,
     &         SCAL, E1H, E2H, Z1, Z2, Z3, F1, F2, F3,
     &         KZ1, KZ2, KZ3, MF1, MF2, MF3,
     &         RHS_R, RHS_CX, DF1, DF23, W, CONT, MCONT,
     &         RHS_ERR, ERR_VEC, IPIV_R, IPIV_CX,
     &         0.0D0, 5.0D0, RTOL, ATOL, NSTEP)
      END DO

      CALL CPU_TIME(T1)
      DO ITERS = 1, 50
          DO I = 1, N
              Y(I) = 1.0D0
          END DO
          CALL SOLVE_RADAU(N, N2, Y, M, KMAT, N, N2,
     &         SCAL, E1H, E2H, Z1, Z2, Z3, F1, F2, F3,
     &         KZ1, KZ2, KZ3, MF1, MF2, MF3,
     &         RHS_R, RHS_CX, DF1, DF23, W, CONT, MCONT,
     &         RHS_ERR, ERR_VEC, IPIV_R, IPIV_CX,
     &         0.0D0, 5.0D0, RTOL, ATOL, NSTEP)
      END DO
      CALL CPU_TIME(T2)
      ELAPSED = T2 - T1

      PRINT *, '  Mean time:', ELAPSED/50.0D0*1000.0D0, 'ms'
      PRINT *, '  Total time:', ELAPSED, 's for 50 iters'

      DO I = 1, N
          Y(I) = 1.0D0
      END DO
      CALL SOLVE_RADAU(N, N2, Y, M, KMAT, N, N2,
     &     SCAL, E1H, E2H, Z1, Z2, Z3, F1, F2, F3,
     &     KZ1, KZ2, KZ3, MF1, MF2, MF3,
     &     RHS_R, RHS_CX, DF1, DF23, W, CONT, MCONT,
     &     RHS_ERR, ERR_VEC, IPIV_R, IPIV_CX,
     &     0.0D0, 5.0D0, RTOL, ATOL, NSTEP)
      PRINT *, '  y_final:', Y(1), Y(2), Y(3)
      PRINT *, '  Steps:', NSTEP

      RETURN
      END


C ====================================================================
C Medium tridiagonal system (n=10)
C ====================================================================
      SUBROUTINE BENCH_TRIDIAG10()
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER N, N2
      PARAMETER (N=10, N2=2*N)
      DOUBLE PRECISION Y(N), SCAL(N)
      DOUBLE PRECISION M(N,N), KMAT(N,N)
      DOUBLE PRECISION E1H(N,N), E2H(N2,N2)
      DOUBLE PRECISION Z1(N), Z2(N), Z3(N)
      DOUBLE PRECISION F1(N), F2(N), F3(N)
      DOUBLE PRECISION KZ1(N), KZ2(N), KZ3(N)
      DOUBLE PRECISION MF1(N), MF2(N), MF3(N)
      DOUBLE PRECISION RHS_R(N), RHS_CX(N2)
      DOUBLE PRECISION DF1(N), DF23(N2)
      DOUBLE PRECISION W(N), CONT(N), MCONT(N)
      DOUBLE PRECISION RHS_ERR(N), ERR_VEC(N)
      INTEGER IPIV_R(N), IPIV_CX(N2)
      DOUBLE PRECISION T1, T2, ELAPSED
      INTEGER I, J, ITERS, NSTEP

      ALPHA = 1.0D0
      BET = 0.01D0

      DO I = 1, N
          DO J = 1, N
              M(I,J) = 0.0D0
              KMAT(I,J) = 0.0D0
          END DO
          M(I,I) = 2.0D0
          KMAT(I,I) = ALPHA + 2.0D0*BET
          IF (I .GT. 1) THEN
              M(I,I-1) = 1.0D0
              KMAT(I,I-1) = -BET
          END IF
          IF (I .LT. N) THEN
              M(I,I+1) = 1.0D0
              KMAT(I,I+1) = -BET
          END IF
      END DO

      RTOL = 1.0D-6
      ATOL = 1.0D-8

      DO ITERS = 1, 2
          DO I = 1, N
              Y(I) = 1.0D0
          END DO
          CALL SOLVE_RADAU(N, N2, Y, M, KMAT, N, N2,
     &         SCAL, E1H, E2H, Z1, Z2, Z3, F1, F2, F3,
     &         KZ1, KZ2, KZ3, MF1, MF2, MF3,
     &         RHS_R, RHS_CX, DF1, DF23, W, CONT, MCONT,
     &         RHS_ERR, ERR_VEC, IPIV_R, IPIV_CX,
     &         0.0D0, 1.0D0, RTOL, ATOL, NSTEP)
      END DO

      CALL CPU_TIME(T1)
      DO ITERS = 1, 10
          DO I = 1, N
              Y(I) = 1.0D0
          END DO
          CALL SOLVE_RADAU(N, N2, Y, M, KMAT, N, N2,
     &         SCAL, E1H, E2H, Z1, Z2, Z3, F1, F2, F3,
     &         KZ1, KZ2, KZ3, MF1, MF2, MF3,
     &         RHS_R, RHS_CX, DF1, DF23, W, CONT, MCONT,
     &         RHS_ERR, ERR_VEC, IPIV_R, IPIV_CX,
     &         0.0D0, 1.0D0, RTOL, ATOL, NSTEP)
      END DO
      CALL CPU_TIME(T2)
      ELAPSED = T2 - T1

      PRINT *, '  Mean time:', ELAPSED/10.0D0*1000.0D0, 'ms'
      PRINT *, '  Total time:', ELAPSED, 's for 10 iters'

      DO I = 1, N
          Y(I) = 1.0D0
      END DO
      CALL SOLVE_RADAU(N, N2, Y, M, KMAT, N, N2,
     &     SCAL, E1H, E2H, Z1, Z2, Z3, F1, F2, F3,
     &     KZ1, KZ2, KZ3, MF1, MF2, MF3,
     &     RHS_R, RHS_CX, DF1, DF23, W, CONT, MCONT,
     &     RHS_ERR, ERR_VEC, IPIV_R, IPIV_CX,
     &     0.0D0, 1.0D0, RTOL, ATOL, NSTEP)
      PRINT *, '  Steps:', NSTEP

      RETURN
      END


C ====================================================================
C Large tridiagonal system (n=100)
C ====================================================================
      SUBROUTINE BENCH_TRIDIAG100()
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER N, N2
      PARAMETER (N=100, N2=2*N)
      DOUBLE PRECISION Y(N), SCAL(N)
      DOUBLE PRECISION M(N,N), KMAT(N,N)
      DOUBLE PRECISION E1H(N,N), E2H(N2,N2)
      DOUBLE PRECISION Z1(N), Z2(N), Z3(N)
      DOUBLE PRECISION F1(N), F2(N), F3(N)
      DOUBLE PRECISION KZ1(N), KZ2(N), KZ3(N)
      DOUBLE PRECISION MF1(N), MF2(N), MF3(N)
      DOUBLE PRECISION RHS_R(N), RHS_CX(N2)
      DOUBLE PRECISION DF1(N), DF23(N2)
      DOUBLE PRECISION W(N), CONT(N), MCONT(N)
      DOUBLE PRECISION RHS_ERR(N), ERR_VEC(N)
      INTEGER IPIV_R(N), IPIV_CX(N2)
      DOUBLE PRECISION T1, T2, ELAPSED
      INTEGER I, J, ITERS, NSTEP

      ALPHA = 1.0D0
      BET = 0.01D0

      DO I = 1, N
          DO J = 1, N
              M(I,J) = 0.0D0
              KMAT(I,J) = 0.0D0
          END DO
          M(I,I) = 2.0D0
          KMAT(I,I) = ALPHA + 2.0D0*BET
          IF (I .GT. 1) THEN
              M(I,I-1) = 1.0D0
              KMAT(I,I-1) = -BET
          END IF
          IF (I .LT. N) THEN
              M(I,I+1) = 1.0D0
              KMAT(I,I+1) = -BET
          END IF
      END DO

      RTOL = 1.0D-6
      ATOL = 1.0D-8

      DO ITERS = 1, 2
          DO I = 1, N
              Y(I) = 1.0D0
          END DO
          CALL SOLVE_RADAU(N, N2, Y, M, KMAT, N, N2,
     &         SCAL, E1H, E2H, Z1, Z2, Z3, F1, F2, F3,
     &         KZ1, KZ2, KZ3, MF1, MF2, MF3,
     &         RHS_R, RHS_CX, DF1, DF23, W, CONT, MCONT,
     &         RHS_ERR, ERR_VEC, IPIV_R, IPIV_CX,
     &         0.0D0, 1.0D0, RTOL, ATOL, NSTEP)
      END DO

      CALL CPU_TIME(T1)
      DO ITERS = 1, 10
          DO I = 1, N
              Y(I) = 1.0D0
          END DO
          CALL SOLVE_RADAU(N, N2, Y, M, KMAT, N, N2,
     &         SCAL, E1H, E2H, Z1, Z2, Z3, F1, F2, F3,
     &         KZ1, KZ2, KZ3, MF1, MF2, MF3,
     &         RHS_R, RHS_CX, DF1, DF23, W, CONT, MCONT,
     &         RHS_ERR, ERR_VEC, IPIV_R, IPIV_CX,
     &     0.0D0, 1.0D0, RTOL, ATOL, NSTEP)
      END DO
      CALL CPU_TIME(T2)
      ELAPSED = T2 - T1

      PRINT *, '  Mean time:', ELAPSED/10.0D0*1000.0D0, 'ms'
      PRINT *, '  Total time:', ELAPSED, 's for 10 iters'

      DO I = 1, N
          Y(I) = 1.0D0
      END DO
      CALL SOLVE_RADAU(N, N2, Y, M, KMAT, N, N2,
     &     SCAL, E1H, E2H, Z1, Z2, Z3, F1, F2, F3,
     &     KZ1, KZ2, KZ3, MF1, MF2, MF3,
     &     RHS_R, RHS_CX, DF1, DF23, W, CONT, MCONT,
     &     RHS_ERR, ERR_VEC, IPIV_R, IPIV_CX,
     &     0.0D0, 1.0D0, RTOL, ATOL, NSTEP)
      PRINT *, '  Steps:', NSTEP

      RETURN
      END


C ====================================================================
C Core RADAU IIA solver (same algorithm as Mojo radau.mojo)
C Solves M*y' = -K*y from T0 to T1
C LDA = leading dimension of M, KMAT, E1H
C LDA2 = leading dimension of E2H
C ====================================================================
      SUBROUTINE SOLVE_RADAU(N, N2, Y, M, KMAT, LDA, LDA2,
     &   SCAL, E1H, E2H, Z1, Z2, Z3, F1, F2, F3,
     &   KZ1, KZ2, KZ3, MF1, MF2, MF3,
     &   RHS_R, RHS_CX, DF1, DF23, W, CONT, MCONT,
     &   RHS_ERR, ERR_VEC, IPIV_R, IPIV_CX,
     &   T0, T1, RTOL, ATOL, NSTEP)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      INTEGER N, N2, LDA, LDA2, NSTEP
      DOUBLE PRECISION Y(N), M(LDA,N), KMAT(LDA,N)
      DOUBLE PRECISION SCAL(N), E1H(LDA,N), E2H(LDA2,N2)
      DOUBLE PRECISION Z1(N), Z2(N), Z3(N)
      DOUBLE PRECISION F1(N), F2(N), F3(N)
      DOUBLE PRECISION KZ1(N), KZ2(N), KZ3(N)
      DOUBLE PRECISION MF1(N), MF2(N), MF3(N)
      DOUBLE PRECISION RHS_R(N), RHS_CX(N2)
      DOUBLE PRECISION DF1(N), DF23(N2)
      DOUBLE PRECISION W(N), CONT(N), MCONT(N)
      DOUBLE PRECISION RHS_ERR(N), ERR_VEC(N)
      INTEGER IPIV_R(N), IPIV_CX(N2)
      DOUBLE PRECISION T0, T1, RTOL, ATOL

      EXTERNAL DGETRF, DGETRS
      INTEGER INFO, NRHS, NEWT, I, J, NIT, NACC
      CHARACTER TRANS
      LOGICAL REJECT, FIRST, CONVERGED, NEWT_FAIL

      DOUBLE PRECISION SQRT6, U1, ALPH, BETA
      DOUBLE PRECISION T11, T12, T13, T21, T22, T23, T31
      DOUBLE PRECISION TI11, TI12, TI13, TI21, TI22, TI23
      DOUBLE PRECISION TI31, TI32, TI33
      DOUBLE PRECISION DD1, DD2, DD3
      DOUBLE PRECISION T, H, H_LU, UROND
      DOUBLE PRECISION SAFETY, FAC1, FAC2, FNEWT
      DOUBLE PRECISION H_OLD, ERR_OLD, THETA
      DOUBLE PRECISION DYNOLD, THQOLD, THETA_LOC
      DOUBLE PRECISION DYO, DYTH, HHFAC, QNEWT
      DOUBLE PRECISION ERR_NORM, ERR_SQ, FAC, QUOT
      DOUBLE PRECISION H_NEW, HACC, ERRACC, FACCON
      DOUBLE PRECISION DNF, K0, POSNEG
      DOUBLE PRECISION FK1, FK2, FK3, WW1, WW2, WW3
      DOUBLE PRECISION THQ, CFAC, FACGUS, S

      SQRT6 = 2.449489742783178D0
      U1 = 3.6378165692476072D0
      ALPH = 2.6753213032678365D0
      BETA = 3.0493570545593676D0
      T11 = 9.1232394870892942792D-02
      T12 = -1.4125529502095420843D-01
      T13 = -3.0029194105147424492D-02
      T21 = 2.4171793270710701896D-01
      T22 = 2.0412935229379993199D-01
      T23 = 3.8294211275726193779D-01
      T31 = 9.6604818261509293619D-01
      TI11 = 4.3255798900631553510D0
      TI12 = 3.3919925181580986954D-01
      TI13 = 5.4177705399358748719D-01
      TI21 = -4.1787185915519047273D0
      TI22 = -3.2768282076106238708D-01
      TI23 = 4.7662355450055045196D-01
      TI31 = -5.0287263494578687595D-01
      TI32 = 2.5719269498556054292D0
      TI33 = -5.9603920482822492497D-01
      DD1 = (-13.0D0 - 7.0D0*SQRT6) / 3.0D0
      DD2 = (-13.0D0 + 7.0D0*SQRT6) / 3.0D0
      DD3 = -1.0D0 / 3.0D0

      UROND = 1.0D-16
      NIT = 7
      SAFETY = 0.9D0
      FAC1 = 0.2D0
      FAC2 = 8.0D0
      FNEWT = MAX(10.0D0*UROND/RTOL, MIN(0.03D0, RTOL**0.5D0))

      T = T0
      POSNEG = 1.0D0
      IF (T1 .LT. T0) POSNEG = -1.0D0

      DO I = 1, N
          SCAL(I) = MAX(ATOL, 1.0D-300) + RTOL * ABS(Y(I))
      END DO

      DNF = 0.0D0
      DO I = 1, N
          K0 = 0.0D0
          DO J = 1, N
              K0 = K0 + KMAT(I,J) * Y(J)
          END DO
          DNF = DNF + (K0/SCAL(I))**2
      END DO
      DNF = SQRT(DNF / DBLE(N))
      IF (DNF .LE. 1.0D-10) THEN
          H = MAX(1.0D-6, ABS(T1-T0)*1.0D-3)
      ELSE
          H = 0.01D0 / DNF
      END IF
      H = MIN(H, ABS(T1-T0))
      H = POSNEG * H

      NSTEP = 0
      NACC = 0
      H_OLD = H
      ERR_OLD = 1.0D-4
      REJECT = .FALSE.
      FIRST = .TRUE.
      THETA = 0.0D0
      HACC = 0.0D0
      ERRACC = 1.0D-2
      H_LU = 0.0D0
      FACCON = 1.0D0

  100 CONTINUE
      IF (POSNEG*(T1-T) .LE. UROND*MAX(ABS(T),ABS(T1)))
     &   RETURN
      IF (NSTEP .GT. 100000) RETURN

      IF (POSNEG*(T+1.01D0*H-T1) .GT. 0.0D0) H = T1 - T
      IF (ABS(H) .LT. 1.0D-14) RETURN

      IF (ABS(H-H_LU) .GT. 1.0D-15*MAX(ABS(H),ABS(H_LU))) THEN
          DO I = 1, N
              DO J = 1, N
                  E1H(I,J) = U1*M(I,J) + H*KMAT(I,J)
              END DO
          END DO
          CALL DGETRF(N, N, E1H, LDA, IPIV_R, INFO)

          DO I = 1, N2
              DO J = 1, N2
                  E2H(I,J) = 0.0D0
              END DO
          END DO
          DO I = 1, N
              DO J = 1, N
                  E2H(I,J) = ALPH*M(I,J) + H*KMAT(I,J)
                  E2H(I,N+J) = -BETA*M(I,J)
                  E2H(N+I,J) = BETA*M(I,J)
                  E2H(N+I,N+J) = ALPH*M(I,J) + H*KMAT(I,J)
              END DO
          END DO
          CALL DGETRF(N2, N2, E2H, LDA2, IPIV_CX, INFO)

          H_LU = H
      END IF

      DO I = 1, N
          SCAL(I) = MAX(ATOL,1.0D-300) + RTOL*ABS(Y(I))
      END DO

      DO I = 1, N
          W(I) = 0.0D0
          DO J = 1, N
              W(I) = W(I) + KMAT(I,J)*Y(J)
          END DO
      END DO

      DO I = 1, N
          Z1(I) = 0.0D0
          Z2(I) = 0.0D0
          Z3(I) = 0.0D0
          F1(I) = 0.0D0
          F2(I) = 0.0D0
          F3(I) = 0.0D0
      END DO

      NEWT_FAIL = .FALSE.
      CONVERGED = .FALSE.
      DYNOLD = 0.0D0
      THQOLD = 0.0D0
      THETA_LOC = ABS(THETA)

      DO NEWT = 1, NIT
          DO I = 1, N
              KZ1(I) = 0.0D0
              KZ2(I) = 0.0D0
              KZ3(I) = 0.0D0
              DO J = 1, N
                  KZ1(I) = KZ1(I) + KMAT(I,J)*Z1(J)
                  KZ2(I) = KZ2(I) + KMAT(I,J)*Z2(J)
                  KZ3(I) = KZ3(I) + KMAT(I,J)*Z3(J)
              END DO
          END DO

          DO I = 1, N
              MF1(I) = 0.0D0
              MF2(I) = 0.0D0
              MF3(I) = 0.0D0
              DO J = 1, N
                  MF1(I) = MF1(I) + M(I,J)*F1(J)
                  MF2(I) = MF2(I) + M(I,J)*F2(J)
                  MF3(I) = MF3(I) + M(I,J)*F3(J)
              END DO
          END DO

          DO I = 1, N
              FK1 = -W(I) - KZ1(I)
              FK2 = -W(I) - KZ2(I)
              FK3 = -W(I) - KZ3(I)
              WW1 = TI11*FK1 + TI12*FK2 + TI13*FK3
              WW2 = TI21*FK1 + TI22*FK2 + TI23*FK3
              WW3 = TI31*FK1 + TI32*FK2 + TI33*FK3
              RHS_R(I) = H*WW1 - U1*MF1(I)
              RHS_CX(I) = H*WW2 - ALPH*MF2(I) + BETA*MF3(I)
              RHS_CX(N+I) = H*WW3 - ALPH*MF3(I) - BETA*MF2(I)
          END DO

          DO I = 1, N
              DF1(I) = RHS_R(I)
          END DO
          TRANS = 'N'
          NRHS = 1
          CALL DGETRS(TRANS, N, NRHS, E1H, LDA, IPIV_R, DF1, N, INFO)

          DO I = 1, N2
              DF23(I) = RHS_CX(I)
          END DO
          CALL DGETRS(TRANS, N2, NRHS, E2H, LDA2, IPIV_CX,
     &                DF23, N2, INFO)

          DYO = 0.0D0
          DO I = 1, N
              DYO = DYO + (DF1(I)/SCAL(I))**2
     &              + (DF23(I)/SCAL(I))**2
     &              + (DF23(N+I)/SCAL(I))**2
          END DO
          DYO = SQRT(DYO / DBLE(3*N))

          IF (NEWT .GT. 1 .AND. NEWT .LT. NIT) THEN
              THQ = DYO / MAX(DYNOLD, UROND)
              IF (NEWT .EQ. 2) THEN
                  THETA_LOC = THQ
              ELSE
                  THETA_LOC = SQRT(THQ*THQOLD)
              END IF
              THQOLD = THQ
              IF (THETA_LOC .LT. 0.99D0) THEN
                  FACCON = THETA_LOC / (1.0D0 - THETA_LOC)
                  DYTH = FACCON * DYO
     &                  * THETA_LOC**(NIT-1-NEWT) / FNEWT
                  IF (DYTH .GE. 1.0D0) THEN
                      QNEWT = MAX(1.0D-4, MIN(20.0D0, DYTH))
                      HHFAC = 0.8D0 * QNEWT
     &                        **(-1.0D0/DBLE(4+NIT-1-NEWT))
                      H = HHFAC * H
                      NEWT_FAIL = .TRUE.
                      GOTO 200
                  END IF
              ELSE
                  NEWT_FAIL = .TRUE.
                  GOTO 200
              END IF
          END IF

          DYNOLD = MAX(DYO, UROND)

          DO I = 1, N
              F1(I) = F1(I) + DF1(I)
              F2(I) = F2(I) + DF23(I)
              F3(I) = F3(I) + DF23(N+I)
              Z1(I) = T11*F1(I) + T12*F2(I) + T13*F3(I)
              Z2(I) = T21*F1(I) + T22*F2(I) + T23*F3(I)
              Z3(I) = T31*F1(I) + 1.0D0*F2(I)
          END DO

          IF (FACCON * DYO .LE. FNEWT) THEN
              CONVERGED = .TRUE.
              GOTO 200
          END IF
      END DO

  200 CONTINUE
      IF (NEWT_FAIL .OR. .NOT. CONVERGED) THEN
          REJECT = .TRUE.
          IF (FIRST) THEN
              H = H * 0.1D0
          ELSE
              H = H * 0.5D0
          END IF
          IF (ABS(H) .LT. 1.0D-14) RETURN
          GOTO 100
      END IF

      THETA = THETA_LOC

      DO I = 1, N
          CONT(I) = DD1*Z1(I) + DD2*Z2(I) + DD3*Z3(I)
      END DO

      DO I = 1, N
          MCONT(I) = 0.0D0
          DO J = 1, N
              MCONT(I) = MCONT(I) + M(I,J)*CONT(J)
          END DO
      END DO

      DO I = 1, N
          RHS_ERR(I) = MCONT(I) - H*W(I)
      END DO

      DO I = 1, N
          ERR_VEC(I) = RHS_ERR(I)
      END DO
      CALL DGETRS(TRANS, N, NRHS, E1H, LDA, IPIV_R,
     &            ERR_VEC, N, INFO)

      ERR_SQ = 0.0D0
      DO I = 1, N
          S = MAX(ATOL,1.0D-300) + RTOL*ABS(Y(I))
          ERR_SQ = ERR_SQ + (ERR_VEC(I)/S)**2
      END DO
      ERR_NORM = SQRT(ERR_SQ / DBLE(N))
      IF (ERR_NORM .LT. 1.0D-10) ERR_NORM = 1.0D-10

      CFAC = SAFETY * DBLE(1 + 2*NIT)
      FAC = MIN(SAFETY, CFAC/DBLE(NEWT+2*NIT))
      QUOT = MAX(1.0D0/FAC2, MIN(1.0D0/FAC1,
     &         ERR_NORM**0.25D0/FAC))
      H_NEW = H / QUOT

      IF (ERR_NORM .LT. 1.0D0) THEN
          FIRST = .FALSE.
          NACC = NACC + 1
          T = T + H
          DO I = 1, N
              Y(I) = Y(I) + Z3(I)
          END DO
          NSTEP = NSTEP + 1

          IF (NACC .GT. 1) THEN
              FACGUS = (HACC/H) * (ERR_NORM**2/ERRACC)**0.25D0
     &                 / SAFETY
              FACGUS = MAX(1.0D0/FAC2, MIN(1.0D0/FAC1, FACGUS))
              QUOT = MAX(QUOT, FACGUS)
              H_NEW = H / QUOT
          END IF

          HACC = H
          ERRACC = MAX(1.0D-2, ERR_NORM)
          H_NEW = POSNEG * MIN(ABS(H_NEW), ABS(T1-T))
          IF (REJECT) H_NEW = POSNEG*MIN(ABS(H_NEW),ABS(H))
          REJECT = .FALSE.
          H = H_NEW
      ELSE
          REJECT = .TRUE.
          IF (FIRST) THEN
              H = H * 0.1D0
          ELSE
              H = H_NEW
          END IF
      END IF

      GOTO 100

      END
