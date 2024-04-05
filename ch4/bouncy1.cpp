// initial version
#include <stdio.h>   // to use printf

int main() {
  double x = 0;       // initialize position
  double y = 2.1;
  double u = 45;     // initialize velocity
  double v = 0;
  double gx = 0;      // initialize acceleration
  double gy = -9.81;
  double dt = 0.04;   // assign time step size
  int ts = 0;         // time step counter

  //show initial values
  printf("%.2f, %.2f, %.2f, %.2f, %.2f\n",ts*dt,x,y,u,v);

  // rewind by half time step for Leapfrog
  u -= 0.5*gx*dt;
  v -= 0.5*gy*dt;
  
  while (y>0) {    // repeat until ground impact
      u += gx*dt; // increment velocity to n+0.5
      v += gy*dt; 
      x += u*dt;  // increment position
      y += v*dt;
      ts++;         // increment time step counter
      // display current position and velocity
      printf("%.2f, %.2f, %.2f, %.2f, %.2f\n",ts*dt,x,y,u,v);
  }
  return 0;
}
