program main
implicit none
integer n
parameter (n=11)
real(kind=8), dimension(n)::x
real(kind=8), dimension(n)::y
integer i

!linspace(-1,1,11)
do i=1,11
  x(i) = -1.0 + 2.0*(i-1)/(n-1)
end do

!evaluate using vector operations
y = max(x*x-2*x,0.)
print "(a7,f6.3,a8,f6.3)","f(-1) =",y(1), ", f(1) =",y(n)
end


