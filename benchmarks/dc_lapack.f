C ******************************************
C     VERSION OF SEPTEMBER 18, 1995      
C ******************************************
C
      SUBROUTINE DECOMR(N,FJAC,LDJAC,FMAS,LDMAS,MLMAS,MUMAS,
     &            M1,M2,NM1,FAC1,E1,LDE1,IP1,IER,IJOB,CALHES,IPHES)
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION FJAC(LDJAC,N),FMAS(LDMAS,NM1),
     &          E1(LDE1,NM1),IP1(NM1)
      COMMON/LINAL/MLE,MUE,MBJAC,MBB,MDIAG,MDIFF,MBDIAG
      GOTO (1,2,3,4,5,6,7,55,55,55,11,12,13,14,15), IJOB
C
C -----------------------------------------------------------
C
   1  CONTINUE
C ---  B=IDENTITY, JACOBIAN A FULL MATRIX
      DO J=1,N
         DO  I=1,N
            E1(I,J)=-FJAC(I,J)
         END DO
         E1(J,J)=E1(J,J)+FAC1
      END DO
      CALL DGETRF (N,N,E1,LDE1,IP1,IER)
      RETURN
C
C -----------------------------------------------------------
C
  11  CONTINUE
C ---  B=IDENTITY, JACOBIAN A FULL MATRIX, SECOND ORDER
      DO J=1,NM1
         JM1=J+M1
         DO I=1,NM1
            E1(I,J)=-FJAC(I,JM1)
         END DO
         E1(J,J)=E1(J,J)+FAC1
      END DO
 45   MM=M1/M2
      DO J=1,M2
         DO I=1,NM1
            SUM=0.D0
            DO K=0,MM-1
               SUM=(SUM+FJAC(I,J+K*M2))/FAC1
            END DO
            E1(I,J)=E1(I,J)-SUM
         END DO
      END DO
      CALL DGETRF (NM1,NM1,E1,LDE1,IP1,IER)
      RETURN
C
C -----------------------------------------------------------
C
   2  CONTINUE
C ---  B=IDENTITY, JACOBIAN A BANDED MATRIX
      DO J=1,N
         DO I=1,MBJAC
            E1(I+MLE,J)=-FJAC(I,J)
         END DO
         E1(MDIAG,J)=E1(MDIAG,J)+FAC1
      END DO
      CALL DGBTRF (N,N,MLE,MUE,E1,LDE1,IP1,IER)
      RETURN
C
C -----------------------------------------------------------
C
  12  CONTINUE
C ---  B=IDENTITY, JACOBIAN A BANDED MATRIX, SECOND ORDER
      DO J=1,NM1
         JM1=J+M1
         DO I=1,MBJAC
            E1(I+MLE,J)=-FJAC(I,JM1)
         END DO
         E1(MDIAG,J)=E1(MDIAG,J)+FAC1
      END DO
  46  MM=M1/M2
      DO J=1,M2
         DO I=1,MBJAC
            SUM=0.D0
            DO K=0,MM-1
               SUM=(SUM+FJAC(I,J+K*M2))/FAC1
            END DO
            E1(I+MLE,J)=E1(I+MLE,J)-SUM
         END DO
      END DO
      CALL DGBTRF (NM1,NM1,MLE,MUE,E1,LDE1,IP1,IER)
      RETURN
C
C -----------------------------------------------------------
C
   3  CONTINUE
C ---  B IS A BANDED MATRIX, JACOBIAN A FULL MATRIX
      DO J=1,N
         DO I=1,N
            E1(I,J)=-FJAC(I,J)
         END DO
         DO I=MAX(1,J-MUMAS),MIN(N,J+MLMAS)
            E1(I,J)=E1(I,J)+FAC1*FMAS(I-J+MBDIAG,J)
         END DO
      END DO
      CALL DGETRF (N,N,E1,LDE1,IP1,IER)
      RETURN
C
C -----------------------------------------------------------
C
  13  CONTINUE
C ---  B IS A BANDED MATRIX, JACOBIAN A FULL MATRIX, SECOND ORDER
      DO J=1,NM1
         JM1=J+M1
         DO I=1,NM1
            E1(I,J)=-FJAC(I,JM1)
         END DO
         DO I=MAX(1,J-MUMAS),MIN(NM1,J+MLMAS)
            E1(I,J)=E1(I,J)+FAC1*FMAS(I-J+MBDIAG,J)
         END DO
      END DO
      GOTO 45
C
C -----------------------------------------------------------
C
   4  CONTINUE
C ---  B IS A BANDED MATRIX, JACOBIAN A BANDED MATRIX
      DO J=1,N
         DO I=1,MBJAC
            E1(I+MLE,J)=-FJAC(I,J)
         END DO
         DO I=1,MBB
            IB=I+MDIFF
            E1(IB,J)=E1(IB,J)+FAC1*FMAS(I,J)
         END DO
      END DO
      CALL DGBTRF (N,N,MLE,MUE,E1,LDE1,IP1,IER)
      RETURN
C
C -----------------------------------------------------------
C
  14  CONTINUE
C ---  B IS A BANDED MATRIX, JACOBIAN A BANDED MATRIX, SECOND ORDER
      DO J=1,NM1
         JM1=J+M1
         DO I=1,MBJAC
            E1(I+MLE,J)=-FJAC(I,JM1)
         END DO
         DO I=1,MBB
            IB=I+MDIFF
            E1(IB,J)=E1(IB,J)+FAC1*FMAS(I,J)
         END DO
      END DO
      GOTO 46
C
C -----------------------------------------------------------
C
   5  CONTINUE
C ---  B IS A FULL MATRIX, JACOBIAN A FULL MATRIX
      DO J=1,N
         DO I=1,N
            E1(I,J)=FMAS(I,J)*FAC1-FJAC(I,J)
         END DO
      END DO
      CALL DGETRF (N,N,E1,LDE1,IP1,IER)
      RETURN
C
C -----------------------------------------------------------
C
  15  CONTINUE
C ---  B IS A FULL MATRIX, JACOBIAN A FULL MATRIX, SECOND ORDER
      DO J=1,NM1
         JM1=J+M1
         DO I=1,NM1
            E1(I,J)=FMAS(I,J)*FAC1-FJAC(I,JM1)
         END DO
      END DO
      GOTO 45
C
C -----------------------------------------------------------
C
   6  CONTINUE
C ---  B IS A FULL MATRIX, JACOBIAN A BANDED MATRIX
C ---  THIS OPTION IS NOT PROVIDED
      RETURN
C
C -----------------------------------------------------------
C
   7  CONTINUE
C ---  B=IDENTITY, JACOBIAN A FULL MATRIX, HESSENBERG-OPTION
      WRITE (6,*) ' HESSENBERG-OPTION IS NOT PROVIDED WITH LAPACK'
      WRITE (6,*) ' USE THE DC_DECSOL.F FILE INSTEAD OF DC_LAPACK.F'
      IER=-10
      RETURN
C
C -----------------------------------------------------------
C
  55  CONTINUE
      RETURN
      END
C
C     END OF SUBROUTINE DECOMR
C
C ***********************************************************
C
C (Simplified version of dc_lapack.f - full file from Hairer's website)
C Note: This file requires LAPACK to compile.
