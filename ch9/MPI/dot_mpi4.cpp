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

  size_t N = 1000000;
  int my_size = N/mpi_size+1;		// number of nodes per processor 
  if (mpi_rank==mpi_size-1) 	// make sure we don't exceed n on the last rank
    my_size = N - mpi_rank*my_size;

	// every rank (process) allocates and initializes this full array
    double *a = new double[my_size];
    double *b = new double[my_size];
  
    //set some values
	int i_start = mpi_rank*my_size; // our first node
    for (size_t i=0;i<my_size;i++) {
		int global_i = i+i_start; // index in the global vector
		a[i] = global_i+1/(double)(N); b[i]=1/(a[i]*N);}

	double dot_local = 0;
	for (size_t i=0; i<my_size; i++) dot_local += a[i]*b[i];

	double dot_global;
	MPI_Reduce(&dot_local,&dot_global,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	if (mpi_rank==0) {
		cout<<"Using "<<mpi_size<<" processors, dot product is "<<dot_global<<endl;
	}
	
	// wrap up and clean up
	MPI_Finalize();
	
    return 0;
}
