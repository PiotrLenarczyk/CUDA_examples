#include <iostream>

using namespace std;
typedef uint32_t uint;
#define H2D cudaMemcpyHostToDevice 
#define D2H cudaMemcpyDeviceToHost

//CPU
uint i = 0;
const uint N = 8E3;
const uint NBytes_f32 = sizeof( float ) * N;
const uint nArrays = 1;                             //single default stream of 1D array
float *h_arr[ nArrays ], *h_result[ nArrays ];      //pinned H2D && D2H transfers

//GPU
float *d_arr[ nArrays ];
const uint nThreads = 512, nBlocks = ( N / nThreads ) + 1;
void initGPUMem( void )
{
    for ( i= 0; i < nArrays; i++ )
    {
        cudaMallocHost( ( void** ) &h_arr[ i ], NBytes_f32 );
        cudaMallocHost( ( void** ) &h_result[ i ], NBytes_f32 );
        cudaMalloc( ( void** ) &d_arr[ i ], NBytes_f32 );
//      ...
//      h_arr[] data load
//      ...        
        cudaMemcpyAsync( &d_arr[ i ], &h_arr[ i ], NBytes_f32, H2D );
    };
};


int freeGPUMem( void )
{
    for ( i= 0; i < nArrays; i++ )
    {
        //HOST
        cudaFreeHost( h_arr[ i ] );
        cudaFreeHost( h_result[ i ] );
        //DEVICE
        cudaFree( d_arr[ i ] );
    };
    cudaDeviceReset();
    return 0;
};

__global__ void emptyKernel( float *d_in )
{
	uint tdx = threadIdx.x + blockIdx.x * blockDim.x;
        printf( "thread[%i].block[%i]\n", tdx, blockDim.x );
};


int main( void )
{
    initGPUMem();
    
    for( i = 0; i < nArrays; i++ )
    {
        emptyKernel<<< 1, 1 >>>( d_arr[ i ] ); //first stream execution
        cudaMemcpyAsync( &h_result[ i ], &d_arr[ i ], NBytes_f32, D2H );
    };
    
	return freeGPUMem();
}
