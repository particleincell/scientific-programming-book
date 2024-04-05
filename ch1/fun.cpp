#include <iostream>

using namespace std;

double fun(double x) {
  double z = x*x - 2*x;
  if (z>=0) return z; else return 0;
}

int main() {
  const int n = 11;
  double y[n];   // using static array
  for (int i=0;i<n;i++) {
    double x = -1.0 + 2.0*i/(n-1);
    y[i] = fun(x);
  }

  cout<<"f(-1)="<<y[0]<<", f(1)="<<y[n-1]<<endl;
  return 0;
}
 
