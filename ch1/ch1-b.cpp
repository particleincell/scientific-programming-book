/*
 C++ version of a tennis ball simulation 
 Brieda, Wang, Martin, "Introduction to Modern Scientific Programming and Numerical Methods", CRC Press 2021
*/
#include <iostream>
#include <fstream>

using namespace std;

// helper function to save results
void saveResults(double *x, double *y, double *u, double *v, double dt, int size) {
  ofstream out("results.csv");

  out<<"t,x,y,u,v\n";
  for (int i=0;i<size;i++) {
    out<<i*dt;                   // time 
    out<<","<<x[i]<<","<<y[i];   // x and y
    out<<","<<u[i]<<","<<v[i];   // u and v
    out<<"\n";                   // new line
  }	
}

int main() {
  // allocate arrays
  const int nn = 100; // number of iterations
  double *x = new double[nn];
  double *y = new double[nn];
  double *vx = new double[nn];
  double *vy = new double[nn];

  // set initial values
  x[0] = 0;       
  y[0] = 2.1;
  vx[0] = 45;     
  vy[0] = 0;

  // additional parameters
  const double gx = 0;      
  const double gy = -9.81;
  const double dt = 0.04;   

  // rewind by half time step for Leapfrog
  vx[0] -= 0.5*gx*dt;
  vy[0] -= 0.5*gy*dt;

  for (int n=0;n<nn;n++) {
    vx[n+1] = vx[n] + gx*dt;  // vx[i+1] is now at t+0.5dt
    vy[n+1] = vy[n] + gy*dt;
    x[n+1] = x[n] + vx[n+1]*dt;
    y[n+1] = y[n] + vy[n+1]*dt;
    if (y[n+1]<0) { // check for impact
        y[n+1] *= -1.0;
        double alpha = 0.5; // bounciness 
        vy[n+1] *= -alpha;
        vx[n+1] *= alpha;  
    }
  }

  // save data using our helper function
  saveResults(x,y,vx,vy,dt,nn);

  // free memory
  delete[] x;
  delete[] y;
  delete[] vx; 
  delete[] vy;

  // normal exit
  return 0;
}

