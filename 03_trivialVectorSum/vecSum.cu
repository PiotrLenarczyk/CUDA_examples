#include <iostream>
#include <vector>
#define N 60000

using namespace std;

__global__ void add( float *a, float *b, float *c )
{
	int tid = blockIdx.x;
	if ( tid < N )
		c[ tid ] = a[ tid ] + b[ tid ];
}

int main ( void )
{
    //host vectors
    vector < float > firstVec( N, 1.11f );
    vector < float > secondVec( N, 3.01f );
    vector < float > resultsVec( N, 0.0f );

	//GPU memory allocation
    float *dev_a, *dev_b, *dev_c;
	cudaMalloc( ( void** )&dev_a, N * sizeof( float ) );
	cudaMalloc( ( void** )&dev_b, N * sizeof( float ) );
	cudaMalloc( ( void** )&dev_c, N * sizeof( float ) );
	
	//copy / download data in direction HostToDevice
	cudaMemcpy( dev_a, &firstVec[0], N * sizeof( float ), cudaMemcpyHostToDevice );
	cudaMemcpy( dev_b, &secondVec[0], N * sizeof( float ), cudaMemcpyHostToDevice );
    
	//calculate vectors sum, using Blocks
	add<<<N,1>>> ( dev_a, dev_b, dev_c );
	
	//copy / upload results data c[] in direction DeviceToHost
	cudaMemcpy( &resultsVec[0], dev_c, N * sizeof( float ), cudaMemcpyDeviceToHost );

	//show results
    for ( int i = 0; i < 5; i++ ) 
        cout << firstVec[ i ] << " + " << secondVec[ i ] << " = " << resultsVec[ i ] << endl;

	//free GPU memory
	cudaFree( dev_a );
	cudaFree( dev_b );
	cudaFree( dev_c );
	
    cudaDeviceReset();
	return 0;
}
