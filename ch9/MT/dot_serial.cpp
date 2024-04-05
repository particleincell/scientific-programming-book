#include <iostream>
using namespace std;
int main() {
  size_t n = 10000000;	//create two large arrays
  double *a = new double[n];
  double *b = new double[n];
  
  //set some values
  for (size_t i=0;i<n;i++) {a[i] = i/(double)(n-1); b[i]=2*a[i];}

  double dot = 0;
  for (size_t i=0;i<n;i++) dot += a[i]*b[i];
  
  cout<<"Serial dot product = "<<dot<<endl;
  
  delete[] a;
  delete[] b; 
  return 0;
}

