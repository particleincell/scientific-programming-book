#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <chrono>
using namespace std;

using type = double;

#define CUDA_ERROR(f) {if (f!=cudaSuccess) {cerr<<cudaGetErrorString(f)<<" on line "<<__LINE__<<endl;exit(-1);}}
__global__ void add(type *a, type *b, type *c, const int N)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if (i<N)
	{
		c[i] = a[i]+b[i];
	}
}
 
int main()
{

	/*allocate vectors*/
	const int N = 50000000;		/*50 million*/
	type *a = new type[N];
	type *b = new type[N];
	type *c = new type[N];
	type *c2 = new type[N];

	/*initialize values*/
	for (int i=0;i<N;i++)
	{
		a[i] = i;
		b[i] = 2*i;
	}

	/*add using CPU*/
	auto start = chrono::high_resolution_clock::now();
	for (int i=0;i<N;i++)
		c[i] = a[i] + b[i];
	auto end = chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end-start;
	cout<<"CPU time "<<duration.count()<<" ms"<<endl;

	type *dev_a;		/*pointer to data on the GPU*/
	type *dev_b;		/*pointer to data on the GPU*/
	type *dev_c;		/*pointer to data on the GPU*/
	int threads_per_block = 512;
	int num_blocks = N/threads_per_block+1;
	CUDA_ERROR(cudaMalloc((void**)&dev_a, N*sizeof(type)));
	CUDA_ERROR(cudaMalloc((void**)&dev_b, N*sizeof(type)));
	CUDA_ERROR(cudaMalloc((void**)&dev_c, N*sizeof(type)));
	
	auto t1 = chrono::high_resolution_clock::now();	
	CUDA_ERROR(cudaMemcpy(dev_a,a,N*sizeof(type),cudaMemcpyHostToDevice));
	CUDA_ERROR(cudaMemcpy(dev_b,b,N*sizeof(type),cudaMemcpyHostToDevice));

	auto t2 = chrono::high_resolution_clock::now();	
	add<<<num_blocks, threads_per_block>>>(dev_a, dev_b, dev_c, N);
	CUDA_ERROR(cudaPeekAtLastError());
	CUDA_ERROR(cudaDeviceSynchronize());
	auto t3 = chrono::high_resolution_clock::now();	
	
	//get results
	CUDA_ERROR(cudaMemcpy(c2,dev_c,N*sizeof(type),cudaMemcpyDeviceToHost));
	auto t4 = chrono::high_resolution_clock::now();	

    std::chrono::duration<double, std::milli> duration1 = t4-t1;
	std::chrono::duration<double, std::milli> duration2 = t3-t2;
	cout<<"GPU time "<<duration2.count()<<" ms"<<endl;
	cout<<"GPU time with memory transfer "<<duration1.count()<<" ms"<<endl;

	/*make sure results agree*/
	bool agree = true;
	for (int i=0;i<N;i++)
		if (c[i]!=c2[i]) {cout<<"Mismatch on i="<<i<<", "<<c[i]<<" "<<c2[i]<<endl;agree=false;break;}

	cout<<"Results "<<(agree?"":"DO NOT ")<<"agree"<<endl;

	delete[] a;
	delete[] b;
	delete[] c;
	delete[] c2;

	return 0;
}

