//%%writefile add.cu
#include "cuda_runtime.h"
#include <iostream>
using namespace std;

/*cuda kernel*/
__global__ void add(float a, float b, float *c)
{
	*c = a + b;
}
 
int main()
{
	float *dev_c;		/*pointer to data on the GPU*/
	cudaMalloc((void**)&dev_c, sizeof(float));
	
	// malloc C version of "new"
	add<<<1,1>>>(1.1,2.2,dev_c);
	
	float c;
	cudaMemcpy(&c,dev_c,sizeof(float),cudaMemcpyDeviceToHost);

	cout<<"GPU computed result is "<<c<<endl;

	cudaFree(dev_c);
	return 0;
}


