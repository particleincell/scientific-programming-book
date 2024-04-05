#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

using namespace std;

#define CUDA_ERROR(f) {if (f!=cudaSuccess) {cerr<<cudaGetErrorString(f)<<" on line "<<__LINE__<<endl;exit(-1);}}
__global__ void add(float *a, float *b, float *c, int N)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if (i<N)
		c[i] = a[i] + b[i];
}
 
int main()
{
	cudaDeviceProp cuda_props;
	CUDA_ERROR(cudaGetDeviceProperties(&cuda_props,0));
	cout<<"Max threads per block = "<<cuda_props.maxThreadsPerBlock<<endl;

	/*allocate vectors*/
	const int N = 10000;		/*number of elements*/
	float *a = new float[N];
	float *b = new float[N];
	float *c = new float[N];

	/*initialize values*/
	for (int i=0;i<N;i++)
	{
		a[i] = i;
		b[i] = 2*i;
	}

	float *dev_a;		/*pointer to data on the GPU*/
	float *dev_b;		/*pointer to data on the GPU*/
	float *dev_c;		/*pointer to data on the GPU*/
	
	CUDA_ERROR(cudaMalloc((void**)&dev_a, N*sizeof(float)));
	CUDA_ERROR(cudaMalloc((void**)&dev_b, N*sizeof(float)));
	CUDA_ERROR(cudaMalloc((void**)&dev_c, N*sizeof(float)));
	
	CUDA_ERROR(cudaMemcpy(dev_a,a,N*sizeof(float),cudaMemcpyHostToDevice));
	CUDA_ERROR(cudaMemcpy(dev_b,b,N*sizeof(float),cudaMemcpyHostToDevice));

	int threads_per_block = 512;
	int num_blocks = (N+threads_per_block-1)/threads_per_block;
	add<<<num_blocks, threads_per_block>>>(dev_a, dev_b, dev_c, N);
	CUDA_ERROR(cudaPeekAtLastError());
	
	//get results
	CUDA_ERROR(cudaMemcpy(c,dev_c,N*sizeof(float),cudaMemcpyDeviceToHost));

	/*make sure the results make sense*/
	for (int i=0;i<10;i++)
		cout<<a[i]<<" + "<<b[i]<<" = "<<c[i]<<endl;

    delete[] a;
	delete[] b;
	delete[] c;

	return 0;
}

