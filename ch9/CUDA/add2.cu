#include <iostream>

using namespace std;

#define CUDA_ERROR(f) {if (f!=cudaSuccess) {cerr<<cudaGetErrorString(f)<<" on line "<<__LINE__<<endl;exit(-1);}}

__global__ void add(float *a, float *b, float *c)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}
 
int main()
{
	/*allocate vectors*/
	const int N = 1000;		/*number of elements*/
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
	
	cudaMalloc((void**)&dev_a, N*sizeof(float));
	cudaMalloc((void**)&dev_b, N*sizeof(float));
	cudaMalloc((void**)&dev_c, N*sizeof(float));
	
	cudaMemcpy(dev_a,a,N*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b,b,N*sizeof(float),cudaMemcpyHostToDevice);

	add<<<1, N>>>(dev_a, dev_b, dev_c);
	//cudaPeekAtLastError();
	
	//get results
	cudaMemcpy(c,dev_c,N*sizeof(float),cudaMemcpyDeviceToHost);

	/*make sure the results make sense*/
	for (int i=0;i<10;i++)
		cout<<a[i]<<" + "<<b[i]<<" = "<<c[i]<<endl;

    delete[] a;
	delete[] b;
	delete[] c;

	return 0;
}

