#include <iostream>
#include <chrono>

using namespace std;
typedef uint32_t uint;
#define H2D cudaMemcpyHostToDevice 
#define D2H cudaMemcpyDeviceToHost
#define OK cudaSuccess

//CPU
uint i = 0, ind = 0;
const uint N = 9 * 1024 * 1024;
const uint NBytes_f32 = sizeof( float ) * N;
const uint nArrays = 1;                             //single default stream of 1D array
float *h_arr[ nArrays ], *h_result[ nArrays ];      //pinned H2D && D2H transfers
float nonPinnedArr[ N ];							//non-pinned H2D && D2H transfers are about 2-3times slower via PCIe

//GPU
float *d_arr[ nArrays ];
__device__ float2 d_arr2[ N / 2 ];
__device__ float3 d_arr3[ N / 3 ];
__device__ float4 d_arr4[ N / 4 ];
const uint nThreads = 512, nBlocks = ( N / nThreads ) + 1;
int freeGPUMem( void )
{
    for ( i= 0; i < nArrays; i++ )
    {
        //HOST
        cudaFreeHost( h_arr[ i ] );
        cudaFreeHost( h_result[ i ] );
        cudaFreeHost( nonPinnedArr );
        //DEVICE
        cudaFree( d_arr[ i ] );
        cudaFree( d_arr2 );
        cudaFree( d_arr3 );
        cudaFree( d_arr4 );
    };
    cudaDeviceReset();
    return 0;
};

void initGPUMem( void )
{
    for ( i= 0; i < nArrays; i++ )
    {
        if ( cudaMallocHost( ( void** ) &h_arr[ i ], NBytes_f32 ) != OK ) { printf( "cudaMallocHost err!\n" ); return; };
        if ( cudaMallocHost( ( void** ) &h_result[ i ], NBytes_f32 ) != OK ) { printf( "cudaMallocHost err!\n" ); return; };
        if ( cudaMalloc( ( void** ) &d_arr, NBytes_f32 ) != OK ) { printf( "cudaMalloc err!\n" ); return; };
//      ...
		for ( ind = 0; ind < N; ind++ )
		{
            h_arr[ i ][ ind ] = float( ind );
            nonPinnedArr[ ind ] = float( ind );
        };
        auto t1 = chrono::high_resolution_clock::now();
        for ( ind = 0; ind < N; ind++ )
            h_arr[ i ][ ind ] += h_arr[ i ][ ind ];
        auto t2 = chrono::high_resolution_clock::now();
        cout << "CPU pinned accesses took <chrono> : "
            << chrono::duration_cast< chrono::nanoseconds >( t2 - t1 ).count()
            << " [ns]\n";
        for ( ind = 0; ind < 3; ind++ )
            cout << "h_arr[" << ind << "]: " << h_arr[ 0 ][ ind ] << endl;
        auto t3 = chrono::high_resolution_clock::now();
        for ( ind = 0; ind < N; ind++ )
            nonPinnedArr[ ind ] += nonPinnedArr[ ind ];
        auto t4 = chrono::high_resolution_clock::now();
        cout << "CPU non-pinned accesses took <chrono> : "
            << chrono::duration_cast< chrono::nanoseconds >( t4 - t3 ).count()
            << " [ns]\n";
//============================================================================
		for ( ind = 0; ind < N; ind++ )
		{
            h_arr[ i ][ ind ] = float( ind );
            nonPinnedArr[ ind ] = float( ind );
        };
		cudaEvent_t start1, stop1;
		cudaEventCreate( &start1 );
		cudaEventCreate( &stop1 );
		cudaEventRecord( start1 );
		for ( ind = 0; ind < N; ind++ )
		    h_arr[ i ][ ind ] += h_arr[ i ][ ind ];
		cudaEventRecord( stop1 );
		cudaEventSynchronize( stop1 );
		float milliseconds = 0.0f;
		cudaEventElapsedTime( &milliseconds, start1, stop1 );
		cout << "CPU pinned accesses took <cudaEvent> : "
		     << milliseconds / float( 1E6 )
		     << " [ns]\n";
		cudaEvent_t start2, stop2;
		cudaEventCreate( &start2 );
		cudaEventCreate( &stop2 );
		cudaEventRecord( start2 );
		for ( ind = 0; ind < N; ind++ )
		    nonPinnedArr[ ind ] += nonPinnedArr[ ind ];
		cudaEventRecord( stop2 );
		cudaEventSynchronize( stop2 );
		float milliseconds = 0.0f;
		cudaEventElapsedTime( &milliseconds, start2, stop2 );
		cout << "CPU non-pinned accesses took <cudaEvent> : "
		     << milliseconds / float( 1E6 )
		     << " [ns]\n";         
//============================================================================        
//CPU/GPU speedups of memory accesses:
//	-worst case scenario of memory utilization: 1th CPU <-> 1th GPU {<chrono>, <cudaEvent>}
//	-typical scenario 1th CPU <-> parallel th GPU {<chrono>, <cudaEvent>}
//	-CPU memory accesses via float4 - check
//	-GDDR5 GPU memory accesses opimalization with tuples of sizes = {1,2,3,4}
//	-both GPU's cudaSetDevice();
//      ...        
		for ( ind = 0; ind < N; ind++ )
            h_arr[ i ][ ind ] = float( ind );
        cudaMemcpy( d_arr[ i ], h_arr[ i ], NBytes_f32, H2D );
    };
};

__global__ void singleThreadAccess( float *d_in )
{
	uint tdx = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tdx < N )
    {
        for ( uint l = 0; l < N; l++ )
            d_in[ l ] += d_in[ l ];
    };
};

__global__ void medianThreadAccess( float *d_in )
{
	uint tdx = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tdx < N )
    {
        d_in[ tdx ] += d_in[ tdx ];
    };
};

__global__ void singleTupleAccess

int main( void )
{
    initGPUMem();
    
    for( i = 0; i < nArrays; i++ )
    {
        auto t1 = chrono::high_resolution_clock::now();
        singleThreadAccess<<< 1, 1 >>>( d_arr[ i ] );
        cudaDeviceSynchronize();
        auto t2 = chrono::high_resolution_clock::now();
        cout << "single thread GPU accesses took <chrono> : "
            << chrono::duration_cast< chrono::nanoseconds >( t2 - t1 ).count()
            << " [ns]\n";
        auto t3 = chrono::high_resolution_clock::now();
		medianThreadAccess<<< nBlocks, nThreads >>>( d_arr[ i ] );
        cudaDeviceSynchronize();
        auto t4 = chrono::high_resolution_clock::now();
        cout << "nBlocks[" << nBlocks << "]; nThreads[" << nThreads << "]; GPU accesses took <chrono> : "
             << chrono::duration_cast< chrono::nanoseconds >( t4 - t3 ).count()
             << " [ns]\n";
//============================================================================  
		cudaEvent_t start1, stop1;
		cudaEventCreate( &start1 );
		cudaEventCreate( &stop1 );                  
        cudaEventRecord( start1 );
        	singleThreadAccess<<< 1, 1 >>>( d_arr[ i ] );
        cudaEventRecord( stop1 );
        cudaEventSynchronize( stop1 );
        float milliseconds = 0.0f;
        cudaEventElapsedTime( &milliseconds, start1, stop1 );
        cout << "single thread GPU accesses took <cudaEvent> : " << milliseconds / float( 1E6 ) << "[ns]\n";
		cudaEvent_t start2, stop2;
		cudaEventCreate( &start2 );
		cudaEventCreate( &stop2 );                  
        cudaEventRecord( start2 );
        	medianThreadAccess<<< nBlocks, nThreads >>>( d_arr[ i ] );
        cudaEventRecord( stop2 );
        cudaEventSynchronize( stop2 );
        milliseconds = 0.0f;
        cudaEventElapsedTime( &milliseconds, start2, stop2 );
        cout << "nBlocks[" << nBlocks << "]; nThreads[" << nThreads << "]; GPU accesses took <cudaEvent> : " 
        	 << milliseconds / float( 1E6 ) << "[ns]\n";
//============================================================================                
        cudaMemcpy( h_result[ i ], d_arr[ i ], NBytes_f32, D2H );
    };
    for ( ind = 0; ind < 3; ind++ )
        cout << "   h_result[" << ind << "]: " << h_result[ 0 ][ ind ] << endl;
    
	return freeGPUMem();
}
