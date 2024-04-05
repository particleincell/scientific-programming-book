# custom function defined in fun.m
function [y] = fun(x)
  z = x^2 - 2*x;  # could also just save as y
  if (z>=0)
    y = z;
  else
    y = 0;
  end
return
  