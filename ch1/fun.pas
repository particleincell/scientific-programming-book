program Fun;
const
  N = 11;

(* custom function *)
function fun(x: real): real;
var
   (* local variable declaration *)
   z: real;
begin
   z := x*x - 2*x;
   if (z >= 0) then
      fun := z
   else
      fun := 0;
end;

(* variable declarations *) 
type
   vec11 = array [0..N] of real;
var
   y: vec11;
   i : integer;
   x : real;

(* main program *) 
begin
  for i :=0 to N do
    begin
      x := -1.0 + 2*i/(N-1);
      y[i] := fun(x);
    end;
  writeln ('y(-1)=',y[0]:5:2,', y(1)=',y[N-1]:5:2);
end.



