#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <chrono>
using namespace std;

using type = float;
#define CUDA_ERROR(f) {if (f!=cudaSuccess) {cerr<<cudaGetErrorString(f)<<" on line "<<__LINE__<<endl;exit(-1);}}
__global__ void add(type *a, type *b, type *c, const int N)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;

	if (i<N)
		c[i] = a[i]+b[i];	
}
 
int main()
{
	cudaDeviceProp cuda_props;

	CUDA_ERROR(cudaGetDeviceProperties(&cuda_props,0));
	if (!cuda_props.deviceOverlap) {
		cerr<<"Your GPU does not support concurrent copy and execute kernels"<<endl;
		return -1;
		}

	/*allocate vectors*/
	const int N = 50000000;		/*50 million*/
	
	type *a;
	type *b;
	type *c;

	/*allocated pinned memory on the CPU*/
	CUDA_ERROR(cudaHostAlloc(&a,sizeof(type)*N,cudaHostAllocDefault));
	CUDA_ERROR(cudaHostAlloc(&b,sizeof(type)*N,cudaHostAllocDefault));
	CUDA_ERROR(cudaHostAlloc(&c,sizeof(type)*N,cudaHostAllocDefault));

	type *dev_a;		/*pointer to data on the GPU*/
	type *dev_b;		/*pointer to data on the GPU*/
	type *dev_c;		/*pointer to data on the GPU*/
	
	/*allocate memory on the GPU*/
	CUDA_ERROR(cudaMalloc((void**)&dev_a, N*sizeof(type)));
	CUDA_ERROR(cudaMalloc((void**)&dev_b, N*sizeof(type)));
	CUDA_ERROR(cudaMalloc((void**)&dev_c, N*sizeof(type)));

	/*compute number of blocks using chunk size*/
	int threads_per_block = 512;
	int num_blocks = (N+threads_per_block-1)/threads_per_block;

	/*starting time for no-stream case*/
	auto t1 = chrono::high_resolution_clock::now();	

	/*copy a and b vectors*/
	CUDA_ERROR(cudaMemcpy(dev_a,a,N*sizeof(type),cudaMemcpyHostToDevice));
	CUDA_ERROR(cudaMemcpy(dev_b,b,N*sizeof(type),cudaMemcpyHostToDevice));
		
	/*launch kernel execution*/
	add<<<num_blocks, threads_per_block>>>(dev_a, dev_b, dev_c, N);
	CUDA_ERROR(cudaPeekAtLastError());

	/*copy back*/
	CUDA_ERROR(cudaMemcpy(c,dev_c,N*sizeof(type),cudaMemcpyDeviceToHost));
	auto t2 = chrono::high_resolution_clock::now();	

    std::chrono::duration<double, std::milli> duration1 = t2-t1;
	cout<<"GPU time with pinned memory "<<duration1.count()<<" ms"<<endl;
	
	bool agree = true;
	for (int i=0;i<N;i++) if (a[i]+b[i]!=c[i]) {agree=false;break;}
	cout<<"Results "<<(agree?"":"DO NOT ")<<"agree"<<endl;

	/*Part 2: streams*/
		
	/*create streams*/
	cudaStream_t stream1,stream2;
	CUDA_ERROR(cudaStreamCreate(&stream1));
	CUDA_ERROR(cudaStreamCreate(&stream2));
	
	const int CHUNK = N/10;
	
	/*recompute number of blocks using chunk size*/
	num_blocks = (CHUNK+threads_per_block-1)/threads_per_block;

	/*starting time*/
	auto t3 = chrono::high_resolution_clock::now();	

	for (int i=0;i<N;i+=CHUNK*2)
	{
		/*schedule copies of chunk of "a" on streams 1 and 2*/
		CUDA_ERROR(cudaMemcpyAsync(dev_a,a+i,CHUNK*sizeof(type),cudaMemcpyHostToDevice,stream1));
		CUDA_ERROR(cudaMemcpyAsync(dev_a,a+i+CHUNK,CHUNK*sizeof(type),cudaMemcpyHostToDevice,stream2));

		/*repeat for vector "b"*/
		CUDA_ERROR(cudaMemcpyAsync(dev_b,b+i,CHUNK*sizeof(type),cudaMemcpyHostToDevice,stream1));
		CUDA_ERROR(cudaMemcpyAsync(dev_b,b+i+CHUNK,CHUNK*sizeof(type),cudaMemcpyHostToDevice,stream2));

		/*schedule kernel execution*/
		add<<<num_blocks, threads_per_block,0,stream1>>>(dev_a, dev_b, dev_c, CHUNK);
		add<<<num_blocks, threads_per_block,0,stream2>>>(dev_a, dev_b, dev_c, CHUNK);
		CUDA_ERROR(cudaPeekAtLastError());

		/*schedule memory copy back*/
		CUDA_ERROR(cudaMemcpyAsync(c+i,dev_c,CHUNK*sizeof(type),cudaMemcpyDeviceToHost,stream1));
		CUDA_ERROR(cudaMemcpyAsync(c+i+CHUNK,dev_c,CHUNK*sizeof(type),cudaMemcpyDeviceToHost,stream2));
	}

	/*wait for code to finish running*/
	CUDA_ERROR(cudaStreamSynchronize(stream1));
	CUDA_ERROR(cudaStreamSynchronize(stream2));

	auto t4 = chrono::high_resolution_clock::now();	

    std::chrono::duration<double, std::milli> duration2 = t4-t3;
	cout<<"GPU time with pinned memory and streams "<<duration2.count()<<" ms"<<endl;

	/*make sure results are correct*/
	agree = true;
	for (int i=0;i<N;i++) if (a[i]+b[i]!=c[i]) {agree=false;break;}
	cout<<"Results "<<(agree?"":"DO NOT ")<<"agree"<<endl;

	/*free CPU memory*/
	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);

	/*free GPU memory*/
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);

	CUDA_ERROR(cudaDeviceReset());

	return 0;
}

