#include <iostream>
#include <math.h>
using namespace std;

constexpr double PI = acos(-1.0);

double fun(double x[2]) {
  return sin((x[0]+0.1)*PI) + (x[0]+0.2)*(x[0]+0.2) -0.5*x[0]*x[1] + x[1]*x[1] + 1;
}

void minimize(double (*f)(double[2]),double x1[2], double x2[2], double nn[2], double &min_val, double min_x[2]) {
  min_val = 1e66;    // some large value
  for (int i=0;i<nn[0];i++) 
    for (int j=0;j<nn[1];j++) {
      double x[2];
      x[0] = x1[0] + i*(x2[0]-x1[0])/(nn[0]-1);
      x[1] = x1[1] + j*(x2[1]-x1[1])/(nn[1]-1);
      double val = f(x);
      if (val<min_val) {
        min_val = val;
        min_x[0]=x[0]; min_x[1]=x[1];      
      }
  }
}

int main() {
 double x1[2] = {-2,-2};
 double x2[2] = {2,2};
 double nn[2] = {100,100};
 double min_val;
 double min_x[2];
 
 minimize(fun,x1,x2,nn,min_val,min_x);
 
 cout<<"Minimum "<<min_val<<" at x=("<<min_x[0]<<","<<min_x[1]<<")"<<endl;

 return 0;
}
