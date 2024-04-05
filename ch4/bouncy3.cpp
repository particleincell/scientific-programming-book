// adds VTP output and organized data into structs
#include <iostream>   // for cout
#include <fstream>    // for ofstream
#include <math.h>     // for sqrt and trig
#include <random>
using namespace std;  // to eliminate std::

std::mt19937 mt_gen{random_device()()};	  // use random device to sample starting seed
std::uniform_real_distribution<double> rnd_dist;  //uniform distribution
double rnd() {return rnd_dist(mt_gen);}

constexpr double PI = acos(-1.0);

struct Sample {   // type name
  double x,y,z;   // position
  double u,v,w;   // velocity
}; 

void writeVTP(Sample *smp, unsigned int num_ts, double dt);


int main() {
  const unsigned int num_ts = 3;  // number of steps
  Sample *smp = new Sample[num_ts];

  smp[0].x = 0;       // initialize position
  smp[0].y = 2.1;
  smp[0].z = 0;
  smp[0].u = 4.5;     // initialize velocity
  smp[0].v = 0;
  smp[0].w = 0;
  const double gx = 0;      // initialize acceleration
  const double gy = -9.81;
  const double gz = 0;
  double dt = 0.02;   // assign time step size

  // rewind by half time step for Leapfrog
  smp[0].u -= 0.5*gx*dt;
  smp[0].v -= 0.5*gy*dt;
  smp[0].w -= 0.5*gz*dt;
  
  for (int ts=0; ts<num_ts-1; ts++) {
      smp[ts+1].u = smp[ts].u + gx*dt; // increment velocity to n+0.5
      smp[ts+1].v = smp[ts].v + gy*dt; 
      smp[ts+1].w = smp[ts].w + gz*dt;
      
      smp[ts+1].x = smp[ts].x + smp[ts].u*dt;  // increment position
      smp[ts+1].y = smp[ts].y + smp[ts].v*dt;
      smp[ts+1].z = smp[ts].z + smp[ts].w*dt;
      
      if (smp[ts+1].y<0) {   // if ground hit
        // compute velocity magnitude
	double mag = sqrt(smp[ts+1].u*smp[ts+1].u + smp[ts+1].v*smp[ts+1].v + smp[ts+1].w*smp[ts+1].w);
	constexpr double a_spec = 0.35;   // specularity coefficient

	double mag_new = 0.8*mag; //inelastic reflection
	if (mag_new<0.05) break;  // terminate once stationary

	// direction for specular reflection
	double spec_i =  smp[ts+1].u/mag;
	double spec_j = -smp[ts+1].v/mag;   // specular reflection
	double spec_k =  smp[ts+1].w/mag;
	
	// pick new diffuse direction
	double sin_theta = sqrt(rnd());
	double cos_theta = sqrt(1-sin_theta*sin_theta);
	double phi = 2*PI*rnd();
	
	double dif_i = sin_theta*cos(phi);  // tangent 1
	double dif_j = cos_theta;           // normal (y) component
	double dif_k = sin_theta*sin(phi);  // tangent 2
	
	// set new velocity
	smp[ts+1].u = (a_spec*spec_i + (1-a_spec)*dif_i)*mag_new;
	smp[ts+1].v = (a_spec*spec_j + (1-a_spec)*dif_j)*mag_new;
	smp[ts+1].w = (a_spec*spec_k + (1-a_spec)*dif_k)*mag_new;
	
        // move to surface hack, should integrate only to impact location
        smp[ts+1].y = 0;
      }
 }

  writeVTP(smp,num_ts,dt);
  cout<<"Done!"<<endl;
  return 0;
}

void writeVTP(Sample *smp, unsigned int num_ts, double dt) {

	// output data
	ofstream out("trace.vtp");            //open output file
	out<<"<?xml version=\"1.0\"?>\n";
	out<<"<VTKFile type=\"PolyData\">\n";
	out<<"<PolyData>\n";
	out<<"<Piece NumberOfPoints=\""<<num_ts<<"\" NumberOfVerts=\"0\" "
	   <<"NumberOfLines=\"1\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n";

	out<<"<Points>\n";
	out<<"<DataArray Name=\"pos\" NumberOfComponents=\"3\" type=\"Float64\" format=\"ascii\">\n";
	for (int k=0;k<num_ts;k++)
		out<<smp[k].x<<" "<<smp[k].y<<" "<<smp[k].z<<"\n";
	out<<"</DataArray>\n";
	out<<"</Points>\n";

	out<<"<Lines>\n";
	out<<"<DataArray Name=\"connectivity\" NumberOfComponents=\"1\" type=\"Int32\" format=\"ascii\">\n";
	for (int k=0;k<num_ts;k++) out<<k<<" ";
	out<<"\n</DataArray>\n";

	out<<"<DataArray Name=\"offsets\" NumberOfComponents=\"1\" type=\"Int32\" format=\"ascii\">\n";
	out<<num_ts<<"\n";
	out<<"</DataArray>\n";
	out<<"</Lines>\n";

	out<<"<PointData>\n";
	out<<"<DataArray Name=\"vel\" NumberOfComponents=\"3\" type=\"Float64\" format=\"ascii\">\n";
	for (int k=0;k<num_ts;k++)
		out<<smp[k].u<<" "<<smp[k].v<<" "<<smp[k].w<<"\n";
	out<<"</DataArray>\n";

	out<<"<DataArray Name=\"time\" NumberOfComponents=\"1\" type=\"Float64\" format=\"ascii\">\n";
	for (int k=0;k<num_ts;k++) 
		out<<k*dt<<" ";
	out<<"\n</DataArray>\n";
	out<<"</PointData>\n";

	out<<"</Piece>\n";
	out<<"</PolyData>\n";
	out<<"</VTKFile>\n";
}
