#include <math.h>
#include <chrono>
#include <iostream>
using namespace std;

int main() {  
  auto start = chrono::high_resolution_clock::now();
  double x = cos(0.1);
  auto end = chrono::high_resolution_clock::now();

  chrono::duration<double> delta = end-start;
  cout<<"Function took "<<delta.count()<<" seconds"<<endl;

  return 0;
}
