/*
Example of launching threads using classes
*/

#include <thread>
#include <iostream>
#include <chrono>
#include <math.h>

using namespace std;

/*wrapper for the calculation code. Each thread belongs to a unique Worker instance*/
class Worker
{
public:
	/*constructor, initializes parameters and launches new thread*/
	Worker(double *a, double *b, double *c, long start, long end):
		a(a), b(b), c(c), start(start), end(end)
	{
		/*initialize finished to false, will be changed once calculation ends*/		
		finished = false;

		/*create new thread*/
		thr = new thread(run, this);
	}

	/*destructor, terminate thread, free memory*/
	~Worker() 
	{
		thr->join();	//rejoin our thread with the main code, kills the thread
		delete thr;		//free memory
	}

	/*thread start*/
	static void run(Worker *p)   {p->add();}

	/*computes c = sqrt(|a*b|) for [start,end)*/
	void add()
	{
		cout<<"Running from "<<start<<" to "<<end<<endl;

		for (long i=start;i<end;i++) 
		  c[i] = a[i] + b[i];
		finished = true;
	}

	bool finished;

protected:
	double *a, *b, *c;
	long start, end;
	thread *thr;
};

/*main function*/
int main(int n_args, char *args[])
{
	int num_threads = 1;	//default value
	unsigned int max_threads = thread::hardware_concurrency();

	/*if arguments passed in*/
	if (n_args>1)
	{
		num_threads = atoi(args[1]);
	}

	cout<<"Running with "<<num_threads<<" of "<<max_threads<<" maximum concurrent threads"<<endl;

	/*allocate vectors*/
	long N = 100000000;

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

	/*data storage (array of pointers to Worker class) for "workers"*/
	Worker **workers = new Worker*[num_threads];	

	long chunk = N/num_threads+1;
	for (int i=0;i<num_threads;i++)
	{
		long start = i*chunk;
		long end = (i+1)*chunk;
		if (end>N) end=N;
		/*example of shared memory - a,b, and c pointers are same for all workers!*/
	    workers[i] = new Worker(a,b,c,start,end);
	}

	/*wait for worker threads to finish*/
 	bool finished;
  	do
  	{
	   finished = true;	   
	   for (int i=0;i<num_threads;i++) {
			finished &= workers[i]->finished;	//boolean and, any false will set finished to false
	   }
	   //sleep for 5 milliseconds
	   this_thread::sleep_for(chrono::milliseconds(5));
	} while (!finished);


	/*capture ending time*/
    auto clock_end = chrono::high_resolution_clock::now();
    cout << "Simulation took "<<chrono::duration_cast<chrono::milliseconds>(clock_end-clock_start).count() << "ms\n";

	/*memory cleanup*/
	delete[] a;
	delete[] b;
	delete[] c;

	/*delete workers, will call destructor*/
	for (int i=0;i<num_threads;i++) {delete workers[i];}
	delete[] workers;

	return 0;
}
