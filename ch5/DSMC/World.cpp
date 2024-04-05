/*defines the simulation domain*/
#include <random>
#include <math.h>
#include <iostream>
#include "World.h"
#include "Field.h"
#include "Species.h"
	
//make an instance of the Rnd class
Rnd rnd;

using namespace std;

/*constructor*/
World::World(int ni, int nj):
	ni{ni}, nj{nj},	nn{ni,nj},
	node_vol(nn), object_id(nn)	{
		time_start =  chrono::high_resolution_clock::now();	//save starting time point
	}

/*sets domain bounding box and computes mesh spacing*/
void World::setExtents(double2 _x0, double2 _xm) {
	/*set origin and the opposite corner*/
	x0 = _x0;
	xm = _xm;

	/*compute spacing by dividing length by the number of cells*/
	for (int i=0;i<2;i++)
		dh[i] = (xm[i]-x0[i])/(nn[i]-1);

	//compute centroid
	xc = 0.5*(x0+xm);

	/*recompute node volumes*/
	computeNodeVolumes();
}

/*returns elapsed wall time in seconds*/
double World::getWallTime() {
  auto time_now = chrono::high_resolution_clock::now();
  chrono::duration<double> time_delta = time_now-time_start;
  return time_delta.count();
}

/*computes node volumes, dx*dy*dz on internal nodes and fractional
 * values on domain boundary faces*/
void World::computeNodeVolumes() {
	for (int i=0;i<ni;i++)
		for (int j=0;j<nj;j++)
			{
				double V = dh[0]*dh[1];			//default volume
				if (i==0 || i==ni-1) V*=0.5;	//reduce by two for each boundary index
				if (j==0 || j==nj-1) V*=0.5;
				node_vol[i][j] = V;
			}
}

/* computes total potential energy from 0.5*eps0*sum(E^2)*/
double World::getPE() {
	return 0;
}

/*sugarcubes a circle centered at (x0,y0)*/
void World::addCircle(const double2 &x0, double radius)
{
    /*save circle parameters*/
    circle_x0 = x0;
    circle_rad2 = radius*radius;

    for (int i=0;i<ni;i++)
        for (int j=0;j<nj;j++)
        {
            /*compute node position*/
            double3 x = pos(i,j);
            if (inCircle(x))
            {
                object_id[i][j] = 1;
            }
        }
}

/*marks k=0 plane as 0V Dirichlet boundary*/
void World::addInlet() {
	for (int j=0;j<nj;j++)
	{
		object_id[0][j] = 2;
	}
}
	
/*returns true if point x is inside or on the circle*/
bool World::inCircle(const double3 &x)
{
	double2 x2 {x[0],x[1]};
	double2 r = x2-circle_x0;	//ray to x

    double r_mag2 = (r[0]*r[0] + r[1]*r[1]);
    if (r_mag2<=circle_rad2) return true;
    return false;
}

/*checks for steady state by comparing change in mass, momentum, and energy*/
bool World::checkSteadyState(vector<Species> &species) {
	// do not do anything if already at steady state
	if (steady_state) return true;

	double tot_mass = 0;
	double tot_mom = 0;
	double tot_en = getPE();
	for (Species &sp:species)
	{
		tot_mass += sp.getRealCount();	//number of real molecules
		double3 mom = sp.getMomentum();
		tot_mom += mag(mom);		//z-component of momentum
		tot_en += sp.getKE();		//add kinetic energy
	}

	/*compute new values to last*/
	const double tol = 1e-4;
	if (abs((tot_mass-last_mass)/tot_mass)<tol &&
		abs((tot_mom-last_mom)/tot_mom)<tol &&
		abs((tot_en-last_en)/tot_en)<tol) {
		steady_state = true;
		cout<<"Steady state reached at time step "<<ts<<endl;
		for (Species &sp:species) sp.clearSamples();   // clear collected data
	}

	/*update prior values*/
	last_mass = tot_mass;
	last_mom = tot_mom;
	last_en = tot_en;
	return steady_state;
}

double World::lineCircleIntersect(const double3 &x1, const double3 &x2)
{
	double2 x1_2d {x1[0],x1[2]};
	double3 B = x2-x1;
	double3 A = x1_2d-circle_x0;
	double a = dot(B,B);
	double b = 2*dot(A,B);
	double c = dot(A,A)-circle_rad2;
	double det = b*b-4*a*c;
	if (det<1e-6) return 0.5;

	double tp = (-b + sqrt(det))/(2*a);
	if (tp<0 || tp>1.0)
	{
		tp = (-b - sqrt(det))/(2*a);
		if (tp<0 || tp>1.0)
		{
			cerr<<"Failed to find a line-circle intersection!"<<endl;
			tp=0.0;	//set as starting point
		}
	}

	return tp;
}

/*returns random diffuse direction sampled at a given surface point*/
double3 World::circleDiffuseVector(const double3 &x)
{
	//pick angles, theta=off normal, psi = azimuthal rotation
	double sin_theta = rnd();
	double cos_theta = sqrt(1-sin_theta*sin_theta);
	double psi = 2*Const::PI*rnd();

	double2 x2d{x[0],x[1]};

	double3 n = unit(x2d-circle_x0);	//normal vector
	double3 t1; //create the first tangent
	if (dot(n,{1,0,0})!=0) t1 = cross(n,{1,0,0});
	else t1 = cross(n,{0,1,0});
	double3 t2 = cross(n,t1); //second tangent

	return sin_theta*cos(psi)*t1+sin_theta*sin(psi)*t2+cos_theta*n;
}

