#include <iostream>
#include "solver.h"

using namespace std;

void gsSolve(Matrix &A, dvector &b, Field &x2d) {
  size_t nu = b.size();
  World &world = x2d.world;
 
  if (nu !=A.nr) throw runtime_error("Mismatched dimensions");
  
  // flatten the x vector
  dvector x(nu);
  for (int j=0;j<world.nj;j++) 
    for (int i=0;i<world.ni;i++) x[world.U(i,j)] = x2d(i,j);
  
  // solver loop
  for (int it=0; it<10000; it++) {  
    for (int r=0; r<A.nr; r++) {
      double dot_nomd = A.dotRow(r,x) - A(r,r)*x[r];    // dot product without the main diagonal component
      double g = (b[r] - dot_nomd)/A(r,r);
      x[r] = x[r] + 1.4*(g-x[r]);           // SOR
    } 
  
    // residue check every 25 iterations
    if (it%25==0) {   
      double sum=0;
      for (int r=0; r<A.nr; r++) {
        double R = b[r] - A.dotRow(r,x);
        sum += R*R;     
      }
     
      double norm = sqrt(sum/nu);
      cout<<"Solver it: "<<it<<", norm: "<<norm<<endl;
      if (norm<1e-4) break;
    }
  } // it
  
  // unpack 1D solution
  for (int j=0; j<world.nj; j++) 
    for (int i=0; i<world.ni; i++) x2d(i,j) = x[world.U(i,j)];
}

