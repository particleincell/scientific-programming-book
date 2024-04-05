#include "Solver.h"

std::vector<double> GSSolver::solve(Matrix &A, std::vector<double> &g) {

	int nr = A.getNr(); 		// number of rows

	std::vector<double> x(nr); // solution vector

    /* solve matrix system */
    for (int it=0; it<10000;it++) { // solver iteration
		for (int r=0;r<nr;r++) {  // loop over rows

			double sum = A.dotRow(r,x) - A(r,r)*x[r];
			double x_star = (g[r] - sum) / A(r,r);  // new estimate for x[r]

			x[r] += 1.4*(x_star-x[r]);     // SOR
        }

		// convergence check, only every 50 iterations
        if (it%50==0) {

			std::vector<double> R = A*x - g;  // residue vector

			// compute average error
			double L2 = sqrt(mag2(R)/nr);

			std::cout<<"solver iteration: "<<it<<", L2 norm: "<<L2<<std::endl;
			if (L2<1e-4) break;  // break out of solver loop
        }
	}

	// return solution vector, hint to compiler to try moving instead of copying
	return std::move(x);
}
