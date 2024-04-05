#include <iostream>
#include <fstream>
using namespace std;

/* Solves a tridiagonal system A=[c|d|e|]*x = y using the Thomas Algorithm
   Coefficients of A are modified in place! */
void triDiagSolve(double *c, double *d, double *e, double *y, double *x, size_t n) {
  // forward sweep
  e[0] = e[0]/d[0]; 
  y[0] = y[0]/d[0];
  for (int i=1;i<n-1;i++) {
    e[i] = e[i]/(d[i] - c[i]*e[i-1]);
	y[i] = (y[i] - c[i]*y[i-1])/(d[i]-c[i]*e[i-1]);
  }

  // back substitution
  x[n-1] = y[n-1];		
  for (int i=n-2;i>=0;i--) 
    x[i] = y[i] - e[i]*x[i+1];
}

int main() {
  size_t n = 6;
  double dx = 1.0/(n-1);

  double *c = new double[n];	// left of diagonal
  double *d = new double[n];	// diagonal
  double *e = new double[n];	// right of diagonal

  double *y = new double[n];	// RHS
  double *x = new double[n];	// solution vector

  // clear all data
  for (int i=0;i<n;i++) {
    c[i] = d[i] = e[i] = y[i] = x[i] = 0;
  }

  // set A and d
  for (int i=0;i<n;i++){
    if (i==0 || i==n-1) {
		d[i] = 1.0;
		y[i] = 0.0;
	} else {
		c[i] = 1.0;	 // column to the left
		d[i] = -2.0; // main diagonal
		e[i] = 1.0;	 // column to the right
		y[i] = -100.0*dx*dx; // some forcing values
	}
  }
  
  // solve the system
  triDiagSolve(c,d,e,y,x,n);

  ofstream out("tridiag.csv");  // output file
  out<<"x,f\n";    	// column headers
  for (int i=0;i<n;i++) 
	out<<i*dx<<","<<x[i]<<"\n";

  delete[] c; delete[] d; delete[] e;
  delete[] x; delete[] y;
  return 0;
}
