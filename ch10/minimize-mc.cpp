// monte-carlo search
#include <iostream>
#include <math.h>
#include <random>

using namespace std;

constexpr double PI = acos(-1.0);

double fun(double x[2]) {
  return sin((x[0]+0.1)*PI) + (x[0]+0.2)*(x[0]+0.2) -0.5*x[0]*x[1] + x[1]*x[1] + 1;
}

class Rnd {
public:

	Rnd(): mt_gen{std::random_device()()}, rnd_dist{0,1.0} {}
	double operator() () {return rnd_dist(mt_gen);}

protected:
	std::mt19937 mt_gen;	    //random number generator
	std::uniform_real_distribution<double> rnd_dist;  //uniform distribution
};
Rnd rnd;

void minimize_mc(double (*f)(double[2]),double x1[2], double x2[2], size_t num_samples, double &min_val, double min_x[2]) {
  min_val = 1e66;    // some large value
  for (size_t n=0;n<num_samples;n++) {
    double x[2];
    x[0] = x1[0]+rnd()*(x2[0]-x1[0]);
    x[1] = x1[1]+rnd()*(x2[1]-x1[1]);
	double val = f(x);
    if (val<min_val) {
       min_val = val;
       min_x[0]=x[0]; min_x[1]=x[1];      
    }    
  }
}

int main() {
 double x1[2] = {-2,-2};
 double x2[2] = {2,2};
 constexpr int num_samples = 1000;
 double min_val;
 double min_x[2];
 
 minimize_mc(fun,x1,x2,num_samples,min_val,min_x);
 
 cout<<"Number of samples: "<<num_samples<<endl;
 cout<<"Minimum "<<min_val<<" at x=("<<min_x[0]<<","<<min_x[1]<<")"<<endl;

 return 0;
}
