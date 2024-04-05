/*
C++ heat equation solver
*/
#include <iostream>
#include <math.h>
#include <random>
#include <map>
#include "data.h"
#include "output.h"
#include "solver.h"

// random number generator
class Rnd {
public:
	//constructor: set initial random seed and distribution limits
	Rnd(): mt_gen{std::random_device()()}, rnd_dist{0,1.0} {}
	double operator() () {return rnd_dist(mt_gen);}		//samples in [0,1)
protected:
	std::mt19937 mt_gen;	    //random number generator
	std::uniform_real_distribution<double> rnd_dist;  //uniform distribution
};
Rnd rnd;   // instantiate

using namespace std;

// main
int main() {
  World world(201,201,0.1,0.1);
  Field T(world);
  FieldB fixed(world);
  
  // pick random heat sources
  for (int s=0;s<40;s++) {
    int i = 1+rnd()*(world.ni-2);
    int j = 1+rnd()*(world.nj-2);
    fixed(i,j) = true;
  }
 
  // coefficient matrix and forcing vector
  FiveBandMatrix A(world.nu, world.ni);     
  dvector b(world.nu);		            

 
  // Finite Difference of Laplace equation 
  for (int j=0; j<world.nj; j++)
    for (int i=0; i<world.ni; i++) {
      size_t u = world.U(i,j);
      if (fixed(i,j)) {A(u,u)=1; b[u] = 50+100*rnd(); continue;}
      
      if (i==0) {A(u,u)=1; A(u,u+1)=-1; b[u]=0;}
      else if (i==world.ni-1) {A(u,u)=1; A(u,u-1)=-1; b[u]=0;}
      else if (j==0) {A(u,u)=1; A(u,u+world.ni)=-1; b[u]=0;}
      else if (j==world.nj-1) {A(u,u)=1; A(u,u-world.ni)=-1; b[u]=0;}
      else {  /*standard stencil*/
        A(u,u) = -2/(world.dx*world.dx) - 2/(world.dy*world.dy);
        A(u,u-1) = 1/(world.dx*world.dx);    // T[i-1,j] term
        A(u,u+1) = 1/(world.dx*world.dx);    // T[i+1,j] term
        A(u,u-world.ni) = 1/(world.dy*world.dy);  // T[i,j-1] term
        A(u,u+world.ni) = 1/(world.dy*world.dy);  // T[i,j+1] term
        b[u] = 0;
      }
    }
    
  gsSolve(A, b, T);
  
  // output fields
  map<string,Field*> fields;
  fields["T (K)"] = &T;
  Output::saveVTI(fields, fixed);
  
  cout<<"Done!"<<endl;
  return 0;
}

