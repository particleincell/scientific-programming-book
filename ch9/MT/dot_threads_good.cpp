/*
Incorrect example of a dot product (suffers from race condition)
*/
#include <iostream>
#include <thread>
#include <vector>
using namespace std;

/*dot product function, threads don't work with reference arguments*/
void dot(double *a, double *b, int i1, int i2, double *res) {
  for (int i=i1;i<i2;i++) 
    *res += a[i]*b[i];
}

int main(int num_args, char *args[]) {
    int num_threads = thread::hardware_concurrency();
    if (num_args>1) num_threads = atoi(args[1]);
    cout<<"Running with "<<num_threads<<" threads"<<endl;
    
    size_t N = 1000000;
    double *a = new double[N];
    double *b = new double[N];
    size_t chunk = N/num_threads+1;
      
    //set some values
    for (size_t i=0;i<N;i++) {a[i] = i+1/(double)(N); b[i]=1/(a[i]*N);}

      
    auto start = chrono::high_resolution_clock::now();

	vector<thread> threads(num_threads);
	vector<double> local_dot(num_threads);
	
	for (int i=0;i<num_threads;i++) {
		int i1 = i*chunk;
		int i2 = min((i+1)*chunk,N);
		threads[i] = thread(dot,a,b,i1,i2, &local_dot[i]);
	}

	for (int i=0;i<num_threads;i++)	threads[i].join();
	
	double res = 0;
        for (int i=0;i<num_threads;i++) res+=local_dot[i];

	auto end = chrono::high_resolution_clock::now();

	cout<<"Result Parallel: "<<res<<endl;
  	chrono::duration<double> delta = end-start;
  	cout<<"Calculation took "<<delta.count()<<" seconds"<<endl;

	/*compute serial result*/
	double res_serial=0;
	for (int i=0;i<N;i++) res_serial+=a[i]*b[i];
	cout<<"Result Serial: "<<res_serial<<endl;
	cout<<"Difference: "<<res_serial-res<<endl;

	delete[] a;
	delete[] b;
	return 0;
}

