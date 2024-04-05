#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

using namespace std;

#define CUDA_ERROR(f) {if (f!=cudaSuccess) {cerr<<cudaGetErrorString(f)<<" on line "<<__LINE__<<endl;exit(-1);}}

__global__ void dot(float *a, float *b, float *c, int N)
{
	__shared__ float prod[1024];

	int i = blockIdx.x*blockDim.x+threadIdx.x;

	/*default*/
	prod[threadIdx.x] = 0;

	if (i<N)
		prod[threadIdx.x] = a[i]*b[i];
	
	/*wait for all threads in the block to finish*/
	__syncthreads();

	/*threadIdx starts at zero in each block*/
	if (threadIdx.x==0)
	{
		/*there is a much more efficient way to do this reductions!*/
		float sum=0;
		for (int i=0;i<blockDim.x;i++)
			sum+=prod[i];
		/*save in global memory*/
		c[blockIdx.x] = sum;
	}
}
 
int main()
{
	const int N = 10000;		/*number of elements*/
	const int threads_per_block = 512;
	const int num_blocks = (N+threads_per_block-1)/threads_per_block;

	/*allocate vectors*/
	float *a = new float[N];
	float *b = new float[N];
	

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
	CUDA_ERROR(cudaMalloc((void**)&dev_c, num_blocks*sizeof(float)));
	
	CUDA_ERROR(cudaMemcpy(dev_a,a,N*sizeof(float),cudaMemcpyHostToDevice));
	CUDA_ERROR(cudaMemcpy(dev_b,b,N*sizeof(float),cudaMemcpyHostToDevice));

	dot<<<num_blocks, threads_per_block>>>(dev_a, dev_b, dev_c, N);
	CUDA_ERROR(cudaPeekAtLastError());
	
	//get results
	float *c = new float[num_blocks];
	CUDA_ERROR(cudaMemcpy(c,dev_c,num_blocks*sizeof(float),cudaMemcpyDeviceToHost));

	/*perform final sum on the CPU*/
	float dot_gpu = 0;
	for (int i=0;i<num_blocks;i++) dot_gpu+=c[i];

	/*make sure the results make sense*/
	float dot_cpu=0;
	for (int i=0;i<N;i++)
		dot_cpu+=a[i]*b[i];

	if (fabs((dot_cpu-dot_gpu)/dot_cpu)>1e-4) cout<<"Results disagree! "<<dot_gpu<<" != "<<dot_cpu<<endl;
	else cout<<"Results agree!"<<endl;

    delete[] a;
	delete[] b;
	delete[] c;

	return 0;
}

