/*
Example of adding two vectors using threads
*/

#include <thread>
#include <iostream>
#include <chrono>
#include <math.h>

using namespace std;

/*computes c = a + b for [start,end)*/
void add(double *a, double *b, double *c, long start, long end)
{
	// cout<<"Running from "<<start<<" to "<<end<<endl;
	for (long i=start;i<end;i++) 
	  c[i] = a[i] + b[i];
}

/*main function*/
int main(int n_args, char *args[])
{
  size_t max_threads = thread::hardware_concurrency();
  int num_threads = max_threads;  //default value
 
  /*if arguments passed in*/
  if (n_args>1) {
    num_threads = atoi(args[1]);
  }  

  cout<<"Running with "<<num_threads<<" of "<<max_threads<<" maximum concurrent threads"<<endl;

  /*allocate vectors*/
  long N = 200000000;
  
  cout<<"Attempting to allocate "<<3*sizeof(double)*N/(1024.0*1024.0*1024.0)<<" Gb"<<endl;
  double *a = new double[N];
  double *b = new double[N];
  double *c = new double[N];
  if (a==nullptr || b==nullptr || c==nullptr) {
	  cerr<<"Allocation failed!"<<endl;
	  return -1;
  }

  /*set initial data*/
  for (long i=0;i<N;i++)
  {
	  a[i] = i;
	  b[i] = i%100;
  }
	
  /*grab starting time*/  
  auto clock_start = chrono::high_resolution_clock::now();

  /*array of pointer to threads*/
  thread **threads = new thread*[num_threads];

  long chunk = N/num_threads+1;
  for (int i=0;i<num_threads;i++)
  {
    long start = i*chunk;
    long end = (i+1)*chunk;
    if (end>N) end=N;
    /*example of shared memory - a,b, and c pointers are same for all workers!*/
      threads[i] = new thread(add,a,b,c,start,end);
  }

  /*wait for threads to finish*/
  for (int i=0;i<num_threads;i++)
	threads[i]->join();

  /*capture ending time*/
    auto clock_end = chrono::high_resolution_clock::now();
    cout << "Calculation took "<<chrono::duration_cast<chrono::milliseconds>(clock_end-clock_start).count() << "ms\n";

  /*memory cleanup*/
  delete[] a;
  delete[] b;
  delete[] c;

  /*delete workers, will call destructor*/
  for (int i=0;i<num_threads;i++) {delete threads[i];}
  delete[] threads;

  return 0;
}
