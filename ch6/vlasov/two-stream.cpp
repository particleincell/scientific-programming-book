/*
Example Vlasov code for 2 stream instability
see https://www.particleincell.com/2018/into-to-vlasov-solvers/
Written by Lubos Brieda
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <map>
#include <cstring>
#include <cmath>
#include <vector>

using namespace std;

//storage of domain parameters
struct World
{
	double L;	//length of x
	double u_max;	//extends in the velocity space
	double dx,du;	//cell spacing
	int ni,nj;	//number of nodes in x,y
	bool periodic = true;	//controls whether the world is periodic in x
	
	void setLimits(double L, double u_max) {this->L=L; this->u_max=u_max;}
	void setNodes(int N, int M) {ni=N; nj=2*M-1;dx=L/(ni-1); du=2*u_max/(nj-1);} 
	double getX(int i) {return 0+i*dx;}
	double getU(int j) {return -u_max+j*du;}
	
	//linear interpolation: a higher order scheme needed!
	double interp(double **f, double x, double u)
	{
		double fi = (x-0)/dx;
		double fj = (u-(-u_max))/du;		
		
		//periodic boundaries in i
		if (periodic)
		{
			if (fi<0) fi+=ni-1;		//-0.5 becomes ni-1.5, which is valid since i=0=ni-1
			if (fi>ni-1) fi-=ni-1;
		}
		else if (fi<0 || fi>=ni-1) return 0;
		
		//return zero if velocity less or more than limits
		if (fj<0 || fj>=nj-1) return 0;
		
		int i = (int)fi;
		int j = (int)fj;
		double di = fi-i;
		double dj = fj-j;
		
		double val = (1-di)*(1-dj)*f[i][j];
		if (i<ni-1) val+=(di)*(1-dj)*f[i+1][j];
		if (j<nj-1) val+=(1-di)*(dj)*f[i][j+1];
		if (i<ni-1 && j<nj-1) val+=(di)*(dj)*f[i+1][j+1];
		return val;		
	}
	
	/*makes values on left and right edge identical on periodic systems*/
	void applyBC(double **f)
	{
		if (!periodic)
			return;
		for (int j=0;j<nj;j++)
		{
			f[0][j] = 0.5*(f[0][j]+f[ni-1][j]);
			f[ni-1][j] = f[0][j];
		}
	}
	
	~World() {}
};

//filter to eliminate garbage values like 1e-120
double filter(double a) {if (std::abs(a)<1e-20) return 0; else return a;}

/*saves the provided scalars and vectors to a file*/
void saveVTK(int time_step, World &world, map<string,double**> scalars2D, map<string,double*> scalars1D)
{
	//generate file name
	stringstream ss;
	ss<<"results/2stream";
	if (time_step>=0)
		ss<<"_"<<setw(6)<<setfill('0')<<time_step;
	ss<<".vti";	
	ofstream out(ss.str());

	out<<setprecision(4);

	out<<"<VTKFile type=\"ImageData\">\n";
	out<<"<ImageData Origin=\""<<0<<" "<<-world.u_max<<" "<<0<<"\"";
	out<<" Spacing=\""<<world.dx<<" "<<world.du<<" "<<1<<"\"";
	out<<" WholeExtent=\""<<0<<" "<<world.ni-1<<" "<<0<<" "<<world.nj-1<<" "<<0<<" "<<0<<"\">\n";
	out<<"<Piece Extent=\""<<0<<" "<<world.ni-1<<" "<<0<<" "<<world.nj-1<<" "<<0<<" "<<0<<"\">\n";
	out<<"<PointData>\n";
		
	//user vars, p.first is the string key, p.second is the double* pointer to data
	for (pair<string,double**> p : scalars2D)
	{
		//p.first is the string key, p.second is the double* pointer to data
		out<<"<DataArray Name=\""<<p.first<<"\" NumberOfComponents=\"1\" format=\"ascii\" type=\"Float64\">\n";
		
		for (int j=0;j<world.nj;j++)
		{
			for (int i=0;i<world.ni;i++)
			{
				out<<setprecision(4)<<filter(p.second[i][j])<<" ";					
			}
			out<<"\n";
		}
		out<<"</DataArray>\n";
	}

	for (pair<string,double*> p : scalars1D)
	{
		//p.first is the string key, p.second is the double* pointer to data
		out<<"<DataArray Name=\""<<p.first<<"\" NumberOfComponents=\"1\" format=\"ascii\" type=\"Float64\">\n";
		
		for (int j=0;j<world.nj;j++)
		{
			for (int i=0;i<world.ni;i++)
			{
				out<<setprecision(4)<<filter(p.second[i])<<" ";					
			}
			out<<"\n";
		}
		out<<"</DataArray>\n";
	}

	out<<"</PointData>\n";
	out<<"</Piece>\n";
	out<<"</ImageData>\n";
	out<<"</VTKFile>\n";

	out.close();
}

/*global container to store all allocated memory*/
vector<pair<int,double**>> _vecs2Allocated;
vector<double*> _vecs1Allocated;

/*allocates a 2D [ni][nj] array*/
double** newAndClear(int ni, int nj)
{
	//first allocate a 1D array for the first index
	double **p = new double*[ni];
	for (int i=0;i<ni;i++)
	{
		p[i] = new double[nj];
		memset(p[i],0,sizeof(double)*nj);	
	}
	
	//add to container
	_vecs2Allocated.emplace_back(pair<int,double**>(ni,p));
	
	return p;
}

/*allocates a 1D [ni] array*/
double* newAndClear(int ni)
{
	//first allocate a 1D array for the first index
	double *p = new double[ni];
	memset(p,0,sizeof(double)*ni);	
	
	//add to container
	_vecs1Allocated.emplace_back(p);
	
	return p;
}

/*deletes all memory allocated by newAndClear, gets pointers and sizes from _vecsAllocated*/
void deleteAll()
{
	//for-each loop from C++11, identical to: for (int i=0;i<v.size();i++) {auto &a=v[i];...}
	for (auto &p : _vecs2Allocated)
	{
		//delete the array of pointers
		for (int i=0;i<p.first;i++)
			delete[] p.second[i];
		delete[] p.second;
	}
	
	//also delete 1D arrays
	for (auto &p : _vecs1Allocated)
		delete[] p;	
}

/*solves Poisson's equation with Dirichlet boundaries using the direct Thomas algorithm
and returns the electric field
*/
void solvePoissonsEquationGS(World &world, double *b, double *phi, double *E)
{
	double dx2 = world.dx*world.dx;
	int ni=world.ni;
	double norm;
	double const tol = 1e-3;
	//for (int i=0;i<ni;i++)
	//    phi[i]=0;
	for (int it=0;it<20000;it++)
	{
		/*periodic boundary conditions*/
		phi[0] = 0.5*(phi[ni-2]+phi[1]-dx2*b[0]);
		
		for (int i=1;i<ni-1;i++)
		{
		        
			double g = 0.5*(phi[i-1]+phi[i+1]-dx2*b[i]);
			phi[i] = phi[i] + 1.4*(g-phi[i]);
		}

		phi[ni-1] = 0.5*(phi[ni-2]+phi[1]-dx2*b[ni-1]);
			
		/*check for convergence*/
		if (it%50 == 0)
		{
			double R_sum = 0;
			for (int i=1;i<ni-1;i++)
			{
				double dR =  (phi[i-1]-2.*phi[i]+phi[i+1])/dx2 - b[i];
				R_sum += dR*dR;				
			}
			//periodic boundaries
			double dR = (phi[ni-2]-2*phi[0]+phi[1])/dx2 - b[0];
			R_sum += dR*dR;
			dR = (phi[ni-2]-2*phi[ni-1]+phi[1])/dx2 - b[ni-1];
			R_sum += dR*dR;
			
			norm = sqrt(R_sum/ni);
			//cout<<"  "<<it<<": "<<norm<<endl;
			if (norm<tol)
				break;
		}
	}
	
	if (norm>tol)
		cout<<"GS failed to converge, norm = "<<norm<<endl;
		
	/*set periodic boundary*/	
	phi[0] = 0.5*(phi[0]+phi[ni-1]);
        phi[ni-1] = phi[0];
	
	//compute electric field
	for (int i=1;i<ni-1;i++)
		E[i] = -(phi[i+1]-phi[i-1])/(2*world.dx);
	E[0] = E[ni-1] = -(phi[1]-phi[ni-2])/(2*world.dx);
}

int main()
{
	//constants and parameters
	const double pi = acos(-1.0);		//pi

	//create a variable of type World
	World world;	
	world.setLimits(10,5);
	world.setNodes(401,401);
	int ni = world.ni;
	int nj = world.nj;
	double dx = world.dx;
	double du = world.du;
	double dt = 1/8.0;
	
	cout<<"dx: "<<dx<<" "<<"dv: "<<du<<endl;
			
	double **f = newAndClear(ni,nj); //f
	double **fs = newAndClear(ni,nj); //fs
	double **fss = newAndClear(ni,nj); //fss
	double *ne = newAndClear(ni);	//number density
	double *b = newAndClear(ni);	//Poisson solver RHS, -rho=(ne-1) since ni=1 is assumed
	double *E = newAndClear(ni);	//electric field
	double *phi = newAndClear(ni);	//potential
	
	//map is a list of keys and corresponding values
	map<string,double**> scalars2D; 
	map<string,double*> scalars1D; 
	
	scalars2D["f"] = f;	
	scalars1D["ne"] = ne;
	scalars1D["E"] = E;
	
	world.periodic = true;

	//set initial distribution
	for (int i=0;i<ni;i++)
		for (int j=0;j<nj;j++)
		{
			double x = world.getX(i);
			double u = world.getU(j);			

                        double uth2 = 0.001;  // thermal speed, squared
			double ud1 = 1.6;     // stream 1 drift speed
			double ud2 = -1.3;    // stream 2 drift speed
			double A = (1+0.02*cos(3*pi*x/world.L));
			f[i][j] = 0.5/sqrt(uth2*pi)*exp(-(u-ud1)*(u-ud1)/uth2);
			f[i][j] += 0.5/sqrt(uth2*pi)*exp(-(u-ud2)*(u-ud2)/uth2)*A;
		}
	
	//set some constant e field
	for (int i=0;i<ni;i++)
		E[i] = 0;
	
	int it;
	
	//main loop
	for (it=0;it<=1000;it++)
	{
		//compute f*
		for (int i=0;i<ni;i++)
			for (int j=0;j<nj;j++)
			{
				double u = world.getU(j);
				double x = world.getX(i);
				
				fs[i][j] = world.interp(f,x-u*0.5*dt,u);								
			}
		
		world.applyBC(fs);
				
		//compute number density by integrating f with the trapezoidal rule		
		for (int i=0;i<ni;i++)
		{
			ne[i] = 0;
			for (int j=0;j<nj-1;j++)
				ne[i]+=0.5*(fs[i][j+1]+fs[i][j])*du;
		}
		
		//compute the right hand side, -rho = (ne-1)
		for (int i=0;i<ni;i++)
			b[i] = ne[i]-1;		
		b[0] = 0.5*(b[0]+b[ni-1]);
		b[ni-1] = b[0];

		//solution of the Poisson's equation
		solvePoissonsEquationGS(world,b,phi,E);
		
		//compute f**
		for (int i=0;i<ni;i++)
			for(int j=0;j<nj;j++)
			{
				double u = world.getU(j);
				double x = world.getX(i);
				fss[i][j] = world.interp(fs,x,u+E[i]*dt);				
			}
		
		world.applyBC(fss);
		
		//compute f(n+1)
		for (int i=0;i<ni;i++)
			for(int j=0;j<nj;j++)
			{
				double u = world.getU(j);
				double x = world.getX(i);
				f[i][j] = world.interp(fss,x-u*0.5*dt,u);
			}		
		
		world.applyBC(f);
		
		if (it%100==0)	cout<<it<<endl;
		if (it%5==0)
		      saveVTK(it,world,scalars2D,scalars1D);

	}
				
	saveVTK(it,world,scalars2D,scalars1D);
		
	return 0;
}
