#include <iostream>
#include <chrono>

using namespace std;
typedef uint32_t uint;
#define H2D cudaMemcpyHostToDevice 
#define D2H cudaMemcpyDeviceToHost
#define OK cudaSuccess

//CPU
uint i = 0, ind = 0;
const uint N = 8E6;
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
        if ( cudaMallocHost( ( void** ) &h_arr[ i ], NBytes_f32 ) != OK ) { printf( "cudaMallocHost err!\n" ); return; };
        if ( cudaMallocHost( ( void** ) &h_result[ i ], NBytes_f32 ) != OK ) { printf( "cudaMallocHost err!\n" ); return; };
        if ( cudaMalloc( ( void** ) &d_arr, NBytes_f32 ) != OK ) { printf( "cudaMalloc err!\n" ); return; };
//      ...
        auto t1 = chrono::high_resolution_clock::now();
        for ( ind = 0; ind < N; ind++ )
            h_arr[ i ][ ind ] = float( ind );
        auto t2 = chrono::high_resolution_clock::now();
        cout << "CPU accesses took "
            << chrono::duration_cast< chrono::nanoseconds >( t2 - t1 ).count()
            << " [ns]\n";
        for ( ind = 0; ind < 3; ind++ )
            cout << "h_arr[" << ind << "]: " << h_arr[ 0 ][ ind ] << endl;
//CPU/GPU speedups of memory accesses:
//worst case scenario of memory utilization: 1th CPU <-> 1th GPU
//typical scenario 1th CPU <-> parallel th GPU ( with tuples of sizes = {1,2,3,4} )
//both GPU's
//      ...        
        cudaMemcpy( d_arr[ i ], h_arr[ i ], NBytes_f32, H2D );
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

__global__ void singleThreadAccess( float *d_in )
{
	uint tdx = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tdx < N )
    {
        for ( uint l = 0; l < N; l++ )
            d_in[ l ] = -( float )l;
    };
};

__global__ void medianThreadAccess( float *d_in )
{
	uint tdx = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tdx < N )
    {
        d_in[ tdx ] = -( float )d_in[ tdx ];
    };
};


int main( void )
{
    initGPUMem();
    
    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    
    for( i = 0; i < nArrays; i++ )
    {
        cudaEventRecord( start );
        singleThreadAccess<<< 1, 1 >>>( d_arr[ i ] );
        cudaEventRecord( stop );
        cudaEventSynchronize(stop);
        float milliseconds = 0.0f;
        cudaEventElapsedTime( &milliseconds, start, stop );
        cout << "single thread GPU accesses took " << milliseconds / 1000.0f << "[ns]\n";
        cudaEventRecord( start );
        medianThreadAccess<<< 1, 1 >>>( d_arr[ i ] );
        cudaEventRecord( stop );
        cudaEventSynchronize(stop);
         milliseconds = 0.0f;
        cudaEventElapsedTime( &milliseconds, start, stop );
        cout << "nBlocks[" << nBlocks << "]; nThreads[" << nThreads << "]; GPU accesses took " << milliseconds / 1000.0f << "[ns]\n";
        cudaMemcpy( h_result[ i ], d_arr[ i ], NBytes_f32, D2H );
    };
    for ( ind = 0; ind < 3; ind++ )
        cout << "   h_result[" << ind << "]: " << h_result[ 0 ][ ind ] << endl;
    
	return freeGPUMem();
}
