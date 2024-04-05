10 N = 11
15 DIM Y(N)
20 FOR I = 0 TO N
30 X = -1 + 2*I/(N-1)
50 GOSUB 200
60 Y(I) = YY
70 NEXT I
80 PRINT "Y(-1) = " ;
82 PRINT Y(0) ;
84 PRINT ", Y(1) = " ;
86 PRINT Y(N-1)
90 END

200 REM custom function
    Z = X*X - 2*X
    IF Z >= 0 THEN 
    YY = Z
    ELSE
    YY = 0
    END IF
220 RETURN

