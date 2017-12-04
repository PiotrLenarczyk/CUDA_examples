#include <iostream>
#include <chrono>
#define H2D cudaMemcpyHostToDevice 
#define D2H cudaMemcpyDeviceToHost
#define OK cudaSuccess

using namespace std;
typedef uint32_t uint;

//CPU
uint i = 0, ind = 0;
const uint N = 8E3;
const uint NBytes_f32 = sizeof( float ) * N;
const uint nArrays = 1;                             //single default stream of 1D array
float *h_arr[ nArrays ], *h_result[ nArrays ];      //pinned H2D && D2H transfers

//GPU
float *d_arr[ nArrays ];
__device__ float4 d_sArr[ 1 ];	//d_s[].x;.y;.z;.w; cudaMemcpyToSymbol(*dest,*src,byteSize);cudaMemcpyFromSymbol(*dest,*src,byteSize);
const uint nThreads = 512, nBlocks = ( N / nThreads ) + 1;
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

inline void initGPUMem( void )
{
    for ( i= 0; i < nArrays; i++ )
    {
        if ( cudaMallocHost( ( void** ) &h_arr[ i ], NBytes_f32 ) != cudaSuccess ) { printf( "cudaMallocHost err!\n" ); return; };
        if ( cudaMallocHost( ( void** ) &h_result[ i ], NBytes_f32 ) != cudaSuccess ) { printf( "cudaMallocHost err!\n" ); return; };
        if ( cudaMalloc( ( void** ) &d_arr[ i ], NBytes_f32 ) != cudaSuccess ) { printf( "cudaMalloc err!\n" ); return; };
//      ...
//      h_arr[] data load
            for ( ind = 0; ind < N; ind++ )
                h_arr[ i ][ ind ] = float( ind );
            for ( ind = 0; ind < 3; ind++ )
                cout << "h_arr[" << i << "][" << ind << "]: " << h_arr[ i ][ ind ] << endl;
//      ...        
        cudaMemcpyAsync( d_arr[ i ], h_arr[ i ], NBytes_f32, H2D );
    };
};

__global__ void emptyKernel( float *d_in )
{
	uint tdx = threadIdx.x + blockIdx.x * blockDim.x; 
    if ( tdx < N )
    {
        printf( "thread[%i].block[%i]\n", tdx, blockDim.x );
    };
};

int main( void )
{
    initGPUMem();
    
    for( i = 0; i < nArrays; i++ )
    {
		auto f1 = chrono::high_resolution_clock::now();
        	emptyKernel<<< 1, 1 >>>( d_arr[ i ] );
			cudaDeviceSynchronize();
        auto f2 = chrono::high_resolution_clock::now();
        cout << "GPU kernel took <chrono> : "
            << chrono::duration_cast< chrono::nanoseconds >( f2 - f1 ).count()
            << " [ns]\n"; 
		cudaEvent_t start1, stop1;
		cudaEventCreate( &start1 );
		cudaEventCreate( &stop1 );                  
        cudaEventRecord( start1 );
            emptyKernel<<< 1, 1 >>>( d_arr[ i ] );  
            cudaDeviceSynchronize();
        cudaEventRecord( stop1 );
        cudaEventSynchronize( stop1 );
        float milliseconds = 0.0f;
        cudaEventElapsedTime( &milliseconds, start1, stop1 );
        cout << "nBlocks[" << nBlocks << "]; nThreads[" << nThreads << "]; GPU kernel took <cudaEvent> : " 
        	 << milliseconds * 1000000.0f << "[ns]\n";
        cudaMemcpy( h_result[ i ], d_arr[ i ], NBytes_f32, D2H );
            for ( ind = 0; ind < 3; ind++ )
                cout << "h_result[" << i << "][" << ind << "]: " << h_result[ i ][ ind ] << endl;
    };
    
	return freeGPUMem();
}

