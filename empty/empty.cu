#include <iostream>

using namespace std;
typedef uint32_t uint;
#define H2D cudaMemcpyHostToDevice 
#define D2H cudaMemcpyDeviceToHost
#define OK CUDA_SUCCESS

//CPU
uint i = 0;
const uint N = 8E3;
const uint NBytes_f32 = sizeof( float ) * N;
const uint nArrays = 1;                             //single default stream of 1D array
float *h_arr[ nArrays ], *h_result[ nArrays ];      //pinned H2D && D2H transfers

//GPU
float *d_arr[ nArrays ];
__device__ float4 d_sArr[ 1 ];	//d_s[].x;.y;.z;.w; cudaMemcpyToSymbol(*dest,*src,byteSize);cudaMemcpyFromSymbol(*dest,*src,byteSize);
const uint nThreads = 512, nBlocks = ( N / nThreads ) + 1;
inline void initGPUMem( void )
{
    if ( ( nThreads * nBlocks ) > 16777215 ) { printf( "FPR addressing error!\n" ); return; }; 
    for ( i= 0; i < nArrays; i++ )
    {
        if ( cudaMallocHost( ( void** ) &h_arr[ i ], NBytes_f32 ) != OK ) { printf( "cudaMallocHost err!\n" ); return; };
        if ( cudaMallocHost( ( void** ) &h_result[ i ], NBytes_f32 ) != OK ) { printf( "cudaMallocHost err!\n" ); return; };
        if ( cudaMalloc( ( void** ) &d_arr[ i ], NBytes_f32 ) != OK ) { printf( "cudaMalloc err!\n" ); return; };
//      ...
//      h_arr[] data load
//      ...        
        cudaMemcpyAsync( &d_arr[ i ], &h_arr[ i ], NBytes_f32, H2D );
    };
};


inline int freeGPUMem( void )
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
//   max FPR indexing: nThreads=512; nBlocks=32767; (floatIndex < 16777215 ~=64GB)
	float tdx = threadIdx.x + blockIdx.x * blockDim.x; 
        printf( "thread[%i].block[%i]\n", uint( tdx ), blockDim.x );
};


int main( void )
{
    initGPUMem();
    
    for( i = 0; i < nArrays; i++ )
    {
        emptyKernel<<< 1, 1 >>>( d_arr[ i ] );
        cudaMemcpyAsync( &h_result[ i ], &d_arr[ i ], NBytes_f32, D2H );
    };
    
	return freeGPUMem();
}
