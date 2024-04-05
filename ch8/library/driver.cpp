#include<iostream>
#include<vector>
#include<fstream>
#include<math.h>

#include<map>
#include<string>

#include "Output.h"
#include "Matrix.h"

#include "Solver.h"

using namespace std;

// prototypes


int main() {

	World world;

	int nn = world.nn;
	int ni = world.ni;
	int nj = world.nj;
	double dx = world.dx;
	double dy = world.dy;

	vector<double> g(nn);   // right hand side

	FiveBandMatrix A(nn,ni);   // create a nn*nn banded matrix with ni-offset

    /* set matrix values */
	for (int j=0;j<nj;j++) 
 		for (int i=0;i<ni;i++) {
      		int n = j*ni + i;
  	    	if (i==0 || j==0) { // Dirichlet on imin/jmin
				A(n,n) = 1.0;
	        	g[n] = n/(double)nn;  // some non-uniform boundary
      		} else if (i==ni-1) { // Neumann on imax
				A(n,n) = 1.0;
				A(n,n-1) = -1.0;
				g[n] = 0.0;
			} else if (j==nj-1) {
			    A(n,n) = 1.0/dy;
				A(n,n-ni) = -1.0/dy;
				g[n] = 0.2;		// set non-zero slope			
			}	else {   // regular internal node
				A(n,n-ni) = 1/(dy*dy);
				A(n,n-1) = 1/(dx*dx);
				A(n,n) = -2/(dx*dx) - 2/(dy*dy);
				A(n,n+1) = 1/(dx*dx);
				A(n,n+ni) = 1/(dy*dy);
				g[n] = 0.0;
			}
      	}


	GSSolver solver;
	vector<double> T = solver.solve(A,g);	// T = inv(A)*g
	vector<double> pressure(world.nn);
	vector<double> speed(world.nn);

	map< string , vector<double> > data;

	//data is a container "array" in which the index is a string
	//and the stored item is a vector<double>

	data["temperature"] = T;

	for (int n=0;n<world.nn;n++) pressure[n] = 101250;

	data["P"] = pressure;
	data["speed"] = speed;

	saveVTI("field.vti",world,data);

	// free memory
	return 0;	// normal exit
}  // main end




