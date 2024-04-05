#include <iostream>
#include <sstream>
#include <mpi.h>

using namespace std;

int main(int n_args, char **args) {
	// set up the mpi environment
	MPI_Init(&n_args, &args);

	int mpi_size;
	int mpi_rank;

	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

	// every rank (process) allocates and initializes this full array
	size_t N = 1000000;
    double *a = new double[N];
    double *b = new double[N];
  
    //set some values
    for (size_t i=0;i<N;i++) {a[i] = i+1/(double)(N); b[i]=1/(a[i]*N);}

	int my_size = N/mpi_size+1;		// number of nodes per processor 

	if (mpi_rank==mpi_size-1) 	// make sure we don't exceed n on the last rank
	int my_size = N/mpi_size+1;		// number of nodes per processor 
	int i_start = mpi_rank*my_size; // our first node
	if (mpi_rank==mpi_size-1) 	// make sure we don't exceed n on the last rank
  	  my_size = N - i_start;

	double dot_local = 0;
	for (size_t i=i_start; i<i_start+my_size; i++) dot_local += a[i]*b[i];
	
	{
		// within a block so that ss is a local variable, can reuse name later
		stringstream ss;
		ss<<"Local dot product on rank "<<mpi_rank<<" is "<<dot_local<<endl;
	
		cout<<ss.str();
	}
	
	// wrap up and clean up
	MPI_Finalize();
	
    return 0;
}

