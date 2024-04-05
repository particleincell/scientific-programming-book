#ifndef __SOLVER_H
#define __SOLVER_H

#include <vector>
#include<iostream>
#include<math.h>

#include "Matrix.h"

// solver, receives references to a base Matrix type
class GSSolver {
public:

 std::vector<double> solve(Matrix &A, std::vector<double> &g);

};

#endif
