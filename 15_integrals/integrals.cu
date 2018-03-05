// http://csweb.cs.wfu.edu/bigiron/LittleFE-CUDA-TrapezoidalRule/build/html/cudaAlg.html

// This program implements trapezoidal integration for a function
// f(x) over the interval [c,d] using N subdivisions.  This program
// runs on a host and device (NVIDIA graphics chip with cuda 
// certification).  The function f(x) is implemented as a callable
// function on the device.  The kernel computes the sums f(xi)+f(xi+deltaX).
// The host function computes of the individual sums computed on the
// device and multiplies by deltaX/2.

#include <iostream>
#include <ctime>

using namespace std;
#include <cuda.h>
#include <math_constants.h>
#include <cuda_runtime.h>

// function to integrate, defined as a function on the 
// GPU device
__device__ float myfunction(float a)
{

	return a*a+2.0f*a + 3.0f;
}

// kernel function to compute the summation used in the trapezoidal
// rule for numerical integration
// __global__ __device__ void integratorKernel(float *a, float c, float deltaX, int N)
__global__ void integratorKernel(float *a, float c, float deltaX, int N)
{

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float x = c + (float)idx * deltaX;

	if (idx<N)
	{
		a[idx] = myfunction(x)+myfunction(x+deltaX);

	}


}


// cudaIntegrate() is the host function that sets up the
// computation of the integral of f(x) over the interval
// [c,d].
__host__ float cudaIntegrate(float c, float d, int N)
{
	// deltaX
	float deltaX = (d-c)/N;

	// error code variable
	cudaError_t errorcode = cudaSuccess;

	// size of the arrays in bytes
	int size = N*sizeof(float);

	// allocate array on host and device
	float* a_h = (float *)malloc(size);

	float* a_d;  
	if (( errorcode = cudaMalloc((void **)&a_d,size))!= cudaSuccess)
	   {
		cout << "cudaMalloc(): " << cudaGetErrorString(errorcode) << endl;
		exit(1);
	   }

	// do calculation on device
	int block_size = 256;
	int n_blocks = N/block_size + ( N % block_size == 0 ? 0:1);
	// cout << "blocks: " << n_blocks << endl;
	// cout << "block size: " << block_size << endl;

	integratorKernel <<< n_blocks, block_size >>> (a_d, c, deltaX, N);

	// copy results from device to host
	if((errorcode = cudaMemcpy(a_h, a_d, sizeof(float)*N, cudaMemcpyDeviceToHost))!=cudaSuccess)
	    {
		cout << "cudaMemcpy(): " << cudaGetErrorString(errorcode) << endl;
		exit(1);
	    }


	// add up results
	float sum = 0.0;
	for(int i=0; i<N; i++) sum += a_h[i];
	sum *= deltaX/2.0;

	// clean up
	free(a_h);
	cudaFree(a_d);


	return sum;
}


// utility host function to convert the length of time into 
// micro seconds
__host__ double diffclock(clock_t clock1, clock_t clock2)
{
	double diffticks = clock1-clock2;
	double diffms = diffticks/(CLOCKS_PER_SEC/1000);
	return diffms;
}


// host main program
int main()
{
	clock_t start = clock();
	float answer = cudaIntegrate(0.0,1.0,1000);
	clock_t end = clock();

	cout << "The answer is " << answer << endl;
	cout << "Computation time: " <<  diffclock(end,start);
	cout << "  micro seconds" << endl; 

	return 0;
}