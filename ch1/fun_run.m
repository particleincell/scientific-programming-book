# main script file
n = 11;
y = zeros(n,1);

for i = 1:n
  x = -1 + 2*(i-1)/(n-1);
  y(i) = fun(x);
end
  
fprintf("f(-1)=%g, f(1)=%g\n",y(1),y(n))

x = linspace(-1,1,11)
y = max(x.^2-2*x,0)