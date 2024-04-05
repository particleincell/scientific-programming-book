/* Stochastic gradient descent
*/
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

void minimize_sgd(double (*f)(double[2]), double x_guess[2], double &f0, double x[2]) {
	constexpr int max_iter = 1000;
	constexpr int num_samples = 5;		// number of samples at each point
	
	constexpr double R = 0.001;   // radius in which to pick points
  	constexpr double w = 0.1;      // integration factor
    
  	x[0] = x_guess[0];    // initialize
  	x[1] = x_guess[1];
  	f0 = f(x);
  	double df;    
  	int iter = 0;     

  	do {
	    double x1[2], x2[2];
	  		
	    // point 1
	    double r = (0.01+rnd())*R;		// sample random radius
	    double theta = 2*PI*rnd();
	    x1[0] = x[0] + r*cos(theta);
	    x1[1] = x[1] + r*sin(theta);
	    double df1 = f(x1)-f0;
	
	      // point 2
	      r = rnd()*R;		// sample random radius
	      theta = 2*PI*rnd();
	      x2[0] = x[0] + r*cos(theta);
	      x2[1] = x[1] + r*sin(theta);
	      double df2 = f(x2)-f0;
	     	
	    double a = x1[0]-x[0];		// dx1
	    double b = x1[1]-x[1];		// dy1
	    double c = x2[0]-x[0];		// dx2
	    double d = x2[1]-x[1];		// dy2 	
	    double detA = a*d-b*c;		// det(A)
	     	
	    double invA[2][2] = {{d/detA, -b/detA},{-c/detA, a/detA}}; 	
	    double df_dx = invA[0][0]*df1+invA[0][1]*df2;
	    double df_dy = invA[1][0]*df1+invA[1][1]*df2;
	     	
	    x[0] -= df_dx*w;
	    x[1] -= df_dy*w;      
	     	
	    // calculate difference in f(x) from prior step
	    df = f0;		// save old value
	    f0 = f(x);		// update valule
	    df -= f0;		// difference 	     	
	    cout<<iter<<": f("<<x[0]<<","<<x[1]<<") = "<<f0<<endl;     	     	
	} while (abs(df/f0)>1e-5 && ++iter<max_iter);
}

int main() {
 double x_guess[2] = {-1,-1};
 
 double min_val;
 double min_x[2];
 
 minimize_sgd(fun,x_guess,min_val,min_x);
 
 cout<<"Minimum "<<min_val<<" at x=("<<min_x[0]<<","<<min_x[1]<<")"<<endl;

 return 0;
}
