#include <iostream>
#include <mpi.h>

using namespace std;

int main(int n_args, char **args) {

	// set up the mpi environment
	MPI_Init(&n_args, &args);

	int mpi_size;
	int mpi_rank;

	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

	cout<<"I am "<<mpi_rank<<" of "<<mpi_size<<" running on "<<processor_name<<endl;
	
	// wrap up and clean up
	MPI_Finalize();
	
    return 0;
}

