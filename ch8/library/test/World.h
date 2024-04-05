#ifndef __WORLD_H
#define __WORLD_H

// container to store domain information
struct World {
 	int ni = 81;   // number of nodes
  	int nj = 61;

  	double x0 = 0.1;  // origin
  	double y0 = 0.1;

  	double dx = 0.01;  // cell spacing
  	double dy = 0.01;

  	World() {nn = ni*nj;}  // constructor, just sets number of nodes
  	int nn;
};

#endif
