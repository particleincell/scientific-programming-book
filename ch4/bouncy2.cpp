// adds diffuse reflection and modifies output to use streams
#include <iostream>   // for cout
#include <fstream>    // for ofstream
#include <math.h>     // for sqrt and trig
#include <random>
using namespace std;  // to eliminate std::

std::mt19937 mt_gen{random_device()()};	  // use random device to sample starting seed
std::uniform_real_distribution<double> rnd_dist;  //uniform distribution
double rnd() {return rnd_dist(mt_gen);}

constexpr double PI = acos(-1.0);

int main() {
  double x = 0;       // initialize position
  double y = 2.1;
  double z = 0;
  double u = 4.5;     // initialize velocity
  double v = 0;
  double w = 0;
  double gx = 0;      // initialize acceleration
  double gy = -9.81;
  double gz = 0;
  double dt = 0.02;   // assign time step size
  const int num_ts = 1000;  // number of steps

  ofstream out("results.csv");
  out<<"t,x,y,z,u,v,w\n";
  
  // rewind by half time step for Leapfrog
  u -= 0.5*gx*dt;
  v -= 0.5*gy*dt;
  w -= 0.5*gz*dt;
  
  for (int ts=0; ts<num_ts; ts++) {
      u += gx*dt; // increment velocity to n+0.5
      v += gy*dt; 
      w += gz*dt;
      
      x += u*dt;  // increment position
      y += v*dt;
      z += w*dt;
      
      if (y<0) {   // if ground hit
        // compute velocity magnitude
	double mag = sqrt(u*u + v*v + w*w);
	constexpr double a_spec = 0.35;   // specularity coefficient

	double mag_new = 0.8*mag; //inelastic reflection
	cout<<ts<<" "<<mag_new<<endl;
	if (mag_new<0.05) break;  // terminate once stationary

	// direction for specular reflection
	double spec_i =  u/mag;
	double spec_j = -v/mag;   // specular reflection
	double spec_k =  w/mag;
	
	// pick new diffuse direction
	double sin_theta = sqrt(rnd());
	double cos_theta = sqrt(1-sin_theta*sin_theta);
	double phi = 2*PI*rnd();
	
	double dif_i = sin_theta*cos(phi);  // tangent 1
	double dif_j = cos_theta;           // normal (y) component
	double dif_k = sin_theta*sin(phi);  // tangent 2
	
	// set new velocity
	u = (a_spec*spec_i + (1-a_spec)*dif_i)*mag_new;
	v = (a_spec*spec_j + (1-a_spec)*dif_j)*mag_new;
	w = (a_spec*spec_k + (1-a_spec)*dif_k)*mag_new;
	
        // move to surface hack, should integrate only to impact location
        y = 0;
      }
		
      // write out position and velocity
      out<<ts*dt<<","<<x<<","<<y<<","<<z<<","<<u<<","<<v<<","<<w<<"\n";
  }
  
  cout<<"Done!"<<endl;
  return 0;
}
