#include "Matrix.h"

// vector subtractions, returns c = a-b
std::vector<double> operator-(const std::vector<double> &a, const std::vector<double> &b) {
  int n = a.size();
  std::vector<double> c(n);  // same size as a
  for (int i=0;i<n;i++) c[i] = a[i]-b[i];
  return c;
}

// returns dot product with itself, magnitude squared
double mag2(std::vector<double> &a) {
  double sum = 0;
  for (double d:a) sum+=d*d;
  return sum;
}
