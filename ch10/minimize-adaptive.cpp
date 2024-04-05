#include <iostream>
#include <math.h>
using namespace std;

constexpr double PI = acos(-1.0);

double fun(double x[2]) {
  return sin((x[0]+0.1)*PI) + (x[0]+0.2)*(x[0]+0.2) -0.5*x[0]*x[1] + x[1]*x[1] + 1;
}

double minimize_as(double (*f)(double[2]),double x1[2], double x2[2], double min_val, double min_x[2]) {
  size_t sub_divs = 4;
  
  double dh[2] = {(x2[0]-x1[0])/sub_divs, (x2[1]-x1[1])/sub_divs};
  
  double df;
  for (int i=0;i<sub_divs;i++) 
    for (int j=0;j<sub_divs;j++) {
    
      // evaluate value at centroid
      double x[2];
      x[0] = x1[0] + (i+0.5)*dh[0];
      x[1] = x1[1] + (j+0.5)*dh[1];            
      double val = f(x);
      
      if (val<min_val) {
        df = val-min_val;
        min_val = val;
        min_x[0]=x[0]; min_x[1]=x[1];              
      }
  }
  
  cout<<"f("<<min_x[0]<<","<<min_x[1]<<") = "<<min_val<<endl;
  if (abs(df)>0.001) {
    for (int i=0;i<2;i++) {
       x1[i] = min_x[i]-0.5*dh[i];
       x2[i] = min_x[i]+0.5*dh[i];
    }
    return minimize_as(f,x1,x2,min_val,min_x);    
  }
  
  return min_val;
}

double minimize_as(double (*f)(double[2]),double x1[2], double x2[2], double min_x[2]) {
  return minimize_as(f,x1,x2,1e66,min_x);
}


int main() {
 double x1[2] = {-2,-2};
 double x2[2] = {2,2};
 double min_val;
 double min_x[2];
 
 min_val = minimize_as(fun,x1,x2,min_x);
 
 cout<<"Minimum "<<min_val<<" at x=("<<min_x[0]<<","<<min_x[1]<<")"<<endl;

 return 0;
}
