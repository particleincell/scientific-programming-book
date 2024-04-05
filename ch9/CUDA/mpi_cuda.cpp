#include <mpi.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>

using namespace std;

#define CUDA_ERROR(f) {if (f!=cudaSuccess) {cerr<<cudaGetErrorString(f)<<" on line "<<__LINE__<<endl;exit(-1);}}

/*cuda kernel*/
__global__ void add(float a, float b, float *c)
{
        *c = a + b;
}

int main(int argc, char **argv)
{
  //Initialize MPI
  MPI_Init(NULL, NULL);

  // Get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  float *dev_c;         /*pointer to data on the GPU*/
  CUDA_ERROR(cudaMalloc((void**)&dev_c, sizeof(float)));

  add<<<1,1>>>(1.1,2.2,dev_c);
  CUDA_ERROR(cudaPeekAtLastError());

  float c;
  CUDA_ERROR(cudaMemcpy(&c,dev_c,sizeof(float),cudaMemcpyDeviceToHost));

  // Pirnt a message
  cout<<"Processor "<<processor_name<<" used GPU to compute "<<c<<endl;

  // Finalize MPI
  MPI_Finalize();

  CUDA_ERROR(cudaFree(dev_c));
  return 0;
}
