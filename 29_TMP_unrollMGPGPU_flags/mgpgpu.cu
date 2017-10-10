#include <iostream>

using namespace std;

//CPU
typedef unsigned int uint;
int i = 0;
//GPU
cudaDeviceProp gpuProperties;
const uint N = 1E8;
const uint nThreads = 512;
const uint nBlocks = ( N / nThreads ) + 1;
const uint unrolling = 32;
__device__ float d_f[ N ]; 
__global__ void loop();
__global__ void unrollLoop();
void freeGPU()
{
    cudaFree( d_f );
    cudaDeviceReset();
}

int main( void )
{
    int gpuCount = 0;
    cudaGetDeviceCount( &gpuCount );
    for ( i = 0; i < gpuCount; i++ )
    {
        cudaSetDevice( i );
        cudaGetDeviceProperties( &gpuProperties, i );
        cout << gpuProperties.name << ": " << endl;
        loop<<< 1, 1 >>>();
    }
    
    
    freeGPU();
    return 0;
}

__global__ void loop()
{
    uint iter = 0;
    for ( iter = 0; iter < N; iter++ )
    {
        d_f[ iter ] = 0.0f;
    }
}
