
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "main.h"

//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}


cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {

        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {

        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {

        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {

        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {

        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {

        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);


    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {

        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {

        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}


int test()
{
	int count;
	cudaGetDeviceCount(&count);
	printf("%d\n",count);
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
       
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
       
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.

__global__ void adda(int count,int* a,int *b,int* c)
{
	
	int i= blockDim.x* blockIdx.x +threadIdx.x;
	if(i<count)
		c[i]=a[i]+b[i];
}

void launch(int* a,int* b,int* c,int testsize)
{
	int threadsize=256;
	int blocksize=256;

	adda<<<blocksize,threadsize>>>(testsize,a,b,c);

	cudaDeviceSynchronize();
}

int main()
{
	int testsize=100000;
	int* a;
	int* b;

	//a=new int[testsize];
	//b=new int[testsize];

	cudaMallocManaged(&a,sizeof(int)*testsize);
	cudaMallocManaged(&b,sizeof(int)*testsize);

	for (int i = 0; i < testsize; i++)
	{
		a[i]=i;
		b[i]=testsize-i;
	}
	int* c;

	//c=new int[testsize];
	cudaMallocManaged((void**)&c,sizeof(int)*testsize);
	launch(a,b,c,testsize);

	for (int i = 0; i < testsize; i++)
	{
		if(testsize%10==0)
			printf("%d ",c[i]);
			
	}

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);

	cudaDeviceReset;
    return 0;
}

__global__ void printValue( int *value) {
	++value[blockIdx.x];
}
 
void hostFunction(int *value){
 
	value[0]=1;
	value[1]=2;
	printValue<<< 2, 1 >>>(value);
	cudaDeviceSynchronize();
	cudaFree(value);
}
 
int amain() {
	int *value;
	cudaMallocManaged(&value, 2 * sizeof(int));

	cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {

       // goto Error;
    }
	hostFunction(value);
	return 0;
}



