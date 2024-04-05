c23456*****************************************************************x
      PROGRAM MAIN

      IMPLICIT NONE
      INTEGER N, I
      PARAMETER (N=11)
      REAL*8 Y(N), X, FMAX
      
      DO I = 1, N
        X = -1.0 + 2*(I-1)/(N-1)
        Y(I) = FMAX(X)
      END DO
      WRITE(*,100) Y(1), Y(N)
100   FORMAT('f(-1)=',f6.3,', f(1)=',f6.3)
      STOP
      END

c-----subroutine that evaluates max(x^2-2x,0) 
      REAL*8 FUNCTION FMAX(X)
      IMPLICIT REAL*8 (A-H, O-Z)
      Z = X*X - 2.*X
      IF(z.GT.0) THEN
        FMAX = Z
      ELSE
        FMAX = 0.
      END IF
      RETURN
      END


