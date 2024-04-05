#include <stdio.h>   // to use printf

int main() {
  double x = 0;       // initialize position
  double y = 2.1;
  double vx = 45;     // initialize velocity
  double vy = 0;
  double gx = 0;      // initialize acceleration
  double gy = -9.81;
  double dt = 0.04;   // assign time step size
  int n = 0;          // time step index

  //show initial values
  printf("%.2f, %.2f, %.2f, %.2f, %.2f\n",n*dt,x,y,vx,vy);

  // rewind by half time step for Leapfrog
  vx -= 0.5*gx*dt;
  vy -= 0.5*gy*dt;
  
  while (y>0) {    // repeat until ground impact
      vx += gx*dt; // increment velocity to n+0.5
      vy += gy*dt; 
      x += vx*dt;  // increment position
      y += vy*dt;
      n++;         // increment time step counter
      // display current position and velocity
      printf("%.2f, %.2f, %.2f, %.2f, %.2f\n",n*dt,x,y,vx,vy);
  }
  return 0;
}

