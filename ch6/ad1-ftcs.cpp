/*
FTCS Unsteady Heat Equation solver

*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>

using namespace std;

int main()
{
	//simulation inputs
	double rho = 1;
	double u = 1;
	double D = 0.02;
	double phi0 = 0;
	double phiL = 1;

	//set domain parameters
	double L = 1;
	int ni = 101;
		
	// double x[ni]; 	//static memory allocation
	double *x = new double[ni];	//dynamic memory allocation
	
	//this is basically x = np.linspace(0,L,ni)
	double dx = (L-0)/(ni-1);
	for (int i=0;i<ni;i++)
		x[i] = 0 + dx*i;

	//allocate vectors for analytical solution
	int ni_true = 101;
	double *x_true = new double[ni_true];
	double *phi_true = new double[ni_true];

	//set x vector for the analytical solution
	double dx_true = (L-0)/(ni_true-1);
	for (int i=0;i<ni_true;i++)
	{
		x_true[i] = 0 + dx_true*i;
	}

	//compute the analytical solution
	double Pe = rho*u*L/D;
	for (int i=0;i<ni_true;i++)
		phi_true[i] = phi0 + ((exp(x_true[i]*Pe/L)-1)/(exp(Pe)-1))*(phiL-phi0);
	
	//open output file, assumes "results" folder exists
	ofstream out_true("results/true.csv");
	out_true<<"x,phi_true"<<endl;
	for (int i=0;i<ni_true;i++)
		out_true<<x_true[i]<<","<<phi_true[i]<<"\n";
	out_true.close();
	
	//allocate memory for coefficients, solution, and RHS vectors
	//these will initially contain garbage data
	double *Aw_vec = new double[ni];
	double *Ap_vec = new double[ni];
	double *Ae_vec = new double[ni];
	double *phi = new double[ni];
	double *b = new double[ni];
	
	//dirichlet condition on left
	Ap_vec[0] = 1;		//one on diagonal
	Aw_vec[0] = Ae_vec[0] = 0;	//clear the other two values
	b[0] = phi0;
	
	//dirichlet conditon on right
	Ap_vec[ni-1] = 1;		//one on diagonal
	Aw_vec[ni-1] = Ae_vec[ni-1] = 0;	//clear the other two values
	b[ni-1] = phiL;
	
	//diffusive term
	double AdW = -D/(dx*dx);
	double AdE = -D/(dx*dx);
	double AdP = -(AdE + AdW);

	//convective term
	double AcE = rho*u/(2*dx);
	double AcW = -rho*u/(2*dx);
	double AcP = -(AcW+AcE);

	//contribution from both terms
	double Aw = AdW + AcW;
	double Ap = AdP + AcP;
	double Ae = AdE + AcE;

	//set internal nodes, loop through rows
	for (int i=1;i<ni-1;i++)
	{
		Aw_vec[i] = Aw;
		Ap_vec[i] = Ap;
		Ae_vec[i] = Ae;
	}
	
	//initial values
	for (int i=0;i<ni;i++) 
		phi[i] = 0;		//initialize all phi to 0
	phi[0] = b[0];
	phi[ni-1] = b[ni-1];

	//temporary vector for phi[k+1]
	double *phi_new = new double[ni];	
	
	//iterate using forward time
	double dt = 1e-3;
	
	//integrate solution in time
	for (int it=0;it<100;it++)
	{		
		//set only the non-boundary nodes
		for (int i=1;i<ni-1;i++)
		{
			//compute (A*phi) for node i
			double R = Aw_vec[i]*phi[i-1] + Ap_vec[i]*phi[i] + Ae_vec[i]*phi[i+1];
			phi_new[i] = phi[i] - dt*R;
		}
		
		//copy down
		for (int i=1;i<ni-1;i++)
			phi[i] = phi_new[i];
		
		//open out file in the results folder
		stringstream ss;
		ss<<"results/phi_"<<it<<".csv";		
		ofstream out(ss.str());
		out<<"x,phi"<<endl;
		for (int i=0;i<ni;i++)
			out<<x[i]<<","<<phi[i]<<"\n";
		out.close();		
	}

	//free dynamically allocated memory
	delete[] x;
	delete[] x_true;
	delete[] Aw_vec;
	delete[] Ap_vec;
	delete[] Ae_vec;
	delete[] b;
	delete[] phi;
	delete[] phi_new;
	delete[] phi_true;
	
	return 0;
}
