#include<chrono>
#include<iostream>
using namespace std;
using namespace std::chrono;

int main() {
 size_t N = 4000;
 double **A = new double*[N]; // allocate single array
 for (int r=0;r<N;r++) A[r] = new double[N];
 for (int r=0;r<N;r++) 
   for (int c=0;c<N;c++) A[r][c] = r*c;
	
 auto t1 = high_resolution_clock::now();
 
 // run 500x to improve timing statistics 
 for (int s=0;s<1000;s++) {
    // code to time
    for (int r=0; r<N; r++)   // loop over rows
      for (int c=0; c<N; c++)   // loop over columns
        A[r][c] *= 2.0;
 }
 
  auto t2 = high_resolution_clock::now();
  auto time_span = duration_cast<duration<double>>(t2 - t1);
  cout << "Took " << time_span.count() << " seconds."<<endl;   
  
 for (int r=0;r<N;r++) delete[] A[r];
 delete[] A;
 return 0; 
}
