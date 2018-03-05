#include <iostream>
#include <chrono>

using namespace std;
typedef uint32_t uint;
#define H2D cudaMemcpyHostToDevice 
#define D2H cudaMemcpyDeviceToHost
#define OK cudaSuccess

//CPU
uint i = 0, ind = 0;
//const uint N = 9 * 1024;
const uint N = 9 * 1024 * 1024;
const uint NBytes_f32 = sizeof( float ) * N;
const uint nArrays = 1;                             //single default stream of 1D array
float *h_arr[ nArrays ], *h_result[ nArrays ];      //pinned H2D && D2H transfers
float nonPinnedArr[ N ];							//non-pinned H2D && D2H transfers are about 2-3times slower via PCIe
float2 *h_f2;
float3 *h_f3;
float4 *h_f4;	              						//float4 host ptr ("http://roxlu.com/2013/011/basic-cuda-example")

//GPU
float *d_arr[ nArrays ];
__device__ float2 d_arr2[ N / 2 ];
__device__ float3 d_arr3[ N / 3 ];
__device__ float4 d_arr4[ N / 4 ];
float2 *d_f2;
float3 *d_f3;
float4 *d_f4;							            //float4 device ptr
const uint nThreads = 512, nBlocks = ( N / nThreads ) + 1;
int freeGPUMem( void )
{
    for ( i= 0; i < nArrays; i++ )
    {
        //HOST
        cudaFreeHost( h_arr[ i ] );
        cudaFreeHost( h_result[ i ] );
        cudaFreeHost( nonPinnedArr );
        cudaFree( h_f2 );
        cudaFree( h_f3 );
        cudaFree( h_f4 );
        //DEVICE
        cudaFree( d_arr[ i ] );
        cudaFree( d_arr2 );
        cudaFree( d_arr3 );
        cudaFree( d_arr4 );
        cudaFree( d_f2 );
        cudaFree( d_f3 );
        cudaFree( d_f4 );

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
		     << milliseconds * 1000000.0f
		     << " [ns]\n";
		cudaEvent_t start2, stop2;
		cudaEventCreate( &start2 );
		cudaEventCreate( &stop2 );
		cudaEventRecord( start2 );
		for ( ind = 0; ind < N; ind++ )
		    nonPinnedArr[ ind ] += nonPinnedArr[ ind ];
		cudaEventRecord( stop2 );
		cudaEventSynchronize( stop2 );
		milliseconds = 0.0f;
		cudaEventElapsedTime( &milliseconds, start2, stop2 );
		cout << "CPU non-pinned accesses took <cudaEvent> : "
		     << milliseconds * 1000000.0f
		     << " [ns]\n";       
//============================================================================              
        h_f2 = ( float2* )malloc( NBytes_f32 );
    	for ( ind = 0; ind < N; ind++ )
    	{
    	if ( ( ind % 2 ) == 0 )
            h_f2[ ind / 2 ].x = h_arr[ i ][ ind ];
        else if ( ( ind % 2 ) == 1 )
            h_f2[ ind / 2 ].y = h_arr[ i ][ ind ];
    	};
    	d_f2 = h_f2;
		if ( cudaMalloc( &d_f2, NBytes_f32 ) != OK ) { printf( "cudaMalloc err!" ); return; };
    	if ( cudaMemcpy( d_f2, h_f2, NBytes_f32, H2D ) != OK ) { printf( "cudaMemcpy err!" ); return; };
        h_f3 = ( float3* )malloc( NBytes_f32 );
    	for ( ind = 0; ind < N; ind++ )
    	{
    	if ( ( ind % 3 ) == 0 )
            h_f3[ ind / 3 ].x = h_arr[ i ][ ind ];
        else if ( ( ind % 3 ) == 1 )
            h_f3[ ind / 3 ].y = h_arr[ i ][ ind ];
        else if ( ( ind % 3 ) == 2 )
            h_f3[ ind / 3 ].z = h_arr[ i ][ ind ];
    	};
    	d_f3 = h_f3;
		if ( cudaMalloc( &d_f3, NBytes_f32 ) != OK ) { printf( "cudaMalloc err!" ); return; };
    	if ( cudaMemcpy( d_f3, h_f3, NBytes_f32, H2D ) != OK ) { printf( "cudaMemcpy err!" ); return; };
    	h_f4 = ( float4* )malloc( NBytes_f32 );
    	for ( ind = 0; ind < N; ind++ )
    	{
    	if ( ( ind % 4 ) == 0 )
            h_f4[ ind / 4 ].x = h_arr[ i ][ ind ];
        else if ( ( ind % 4 ) == 1 )
            h_f4[ ind / 4 ].y = h_arr[ i ][ ind ];
        else if ( ( ind % 4 ) == 2 )
            h_f4[ ind / 4 ].z = h_arr[ i ][ ind ];
        else if ( ( ind % 4 ) == 3 )
            h_f4[ ind / 4 ].w = h_arr[ i ][ ind ];
    	};
    	d_f4 = h_f4;
		if ( cudaMalloc( &d_f4, NBytes_f32 ) != OK ) { printf( "cudaMalloc err!" ); return; };
    	if ( cudaMemcpy( d_f4, h_f4, NBytes_f32, H2D ) != OK ) { printf( "cudaMemcpy err!" ); return; };
		for ( ind = 0; ind < N; ind++ )
            h_arr[ i ][ ind ] = float( ind );
        for ( ind = 0; ind < 3; ind++ )
            cout << "h_arr[" << ind << "]: " << h_arr[ 0 ][ ind ] << endl;
        cudaMemcpy( d_arr[ i ], h_arr[ i ], NBytes_f32, H2D );
    };
};

__global__ void nop( const uint N )
{
    uint a = 0;
    for ( size_t i = 0; i < N; i++ );
        a++;
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

__global__ void makeFloat2( float *d_in )
{
	uint tdx = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tdx < N )
    {
        if ( !( tdx % 2 ) )
            d_arr2[ tdx / 2 ].x = d_in[ tdx ];
        else
            d_arr2[ tdx / 2 ].y = d_in[ tdx ];
    };
};

__global__ void float2_Access( void )
{
//     for ( uint i = 0; i < 3; i++ )
//         printf( "d_arr2[%i].x: %f\nd_arr2[%i].y: %f\n", i, d_arr2[ i ].x, i, d_arr2[ i ].y );
    uint tdx = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tdx < ( N / 2 ) )
    {
        d_arr2[ tdx ].x += d_arr2[ tdx ].x;
        d_arr2[ tdx ].y += d_arr2[ tdx ].y;
    };
};

__global__ void makeFloat3( float *d_in )
{
	uint tdx = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tdx < N )
    {
        if ( ( tdx % 3 ) == 0 )
            d_arr3[ tdx / 3 ].x = d_in[ tdx ];
        else if ( ( tdx % 3 ) == 1 )
            d_arr3[ tdx / 3 ].y = d_in[ tdx ];
        else if ( ( tdx % 3 ) == 2 )
            d_arr3[ tdx / 3 ].z = d_in[ tdx ];
    };
};

__global__ void float3_Access( void )
{
//     for ( uint i = 0; i < 2; i++ )
//         printf( "d_arr3[%i].x: %f\nd_arr3[%i].y: %f\nd_arr3[%i].z: %f\n", i, d_arr3[ i ].x, i, d_arr3[ i ].y, i, d_arr3[ i ].z );
    uint tdx = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tdx < ( N / 3 ) )
    {
        d_arr3[ tdx ].x += d_arr3[ tdx ].x;
        d_arr3[ tdx ].y += d_arr3[ tdx ].y;
        d_arr3[ tdx ].z += d_arr3[ tdx ].z;
    };
};

__global__ void makeFloat4( float *d_in )
{
	uint tdx = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tdx < N )
    {
        if ( ( tdx % 4 ) == 0 )
            d_arr4[ tdx / 4 ].x = d_in[ tdx ];
        else if ( ( tdx % 4 ) == 1 )
            d_arr4[ tdx / 4 ].y = d_in[ tdx ];
        else if ( ( tdx % 4 ) == 2 )
            d_arr4[ tdx / 4 ].z = d_in[ tdx ];
        else if ( ( tdx % 4 ) == 3 )
            d_arr4[ tdx / 4 ].w = d_in[ tdx ];
    };
};

__global__ void float4_Access( void )
{
// for ( uint i = 0; i < 2; i++ )
//    printf( "d_inf4[%i].x: %f\nd_inf4[%i].y: %f\nd_inf4[%i].z: %f\nd_inf4[%i].w: %f\n", i, d_inf4[ i ].x, i, d_inf4[ i ].y, i, d_inf4[ i ].z, i, d_inf4[ i ].w );
    uint tdx = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tdx < ( N / 4 ) )
    {
	    d_arr4[ tdx ].x += d_arr4[ tdx ].x;
        d_arr4[ tdx ].y += d_arr4[ tdx ].y;
        d_arr4[ tdx ].z += d_arr4[ tdx ].z;
        d_arr4[ tdx ].w += d_arr4[ tdx ].w;
    };
};

__global__ void arrFloat2_Access( float2 *d_in2 )
{
    uint tdx = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tdx < ( N / 2 ) )
    {
	    d_in2[ tdx ].x += d_in2[ tdx ].x;
        d_in2[ tdx ].y += d_in2[ tdx ].y;
    };
};

__global__ void arrFloat3_Access( float3 *d_in3 )
{
    uint tdx = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tdx < ( N / 3 ) )
    {
	    d_in3[ tdx ].x += d_in3[ tdx ].x;
        d_in3[ tdx ].y += d_in3[ tdx ].y;
        d_in3[ tdx ].z += d_in3[ tdx ].z;
    };
};

__global__ void arrFloat4_Access( float4 *d_inf4 )
{
    uint tdx = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tdx < ( N / 4 ) )
    {
	    d_inf4[ tdx ].x += d_inf4[ tdx ].x;
        d_inf4[ tdx ].y += d_inf4[ tdx ].y;
        d_inf4[ tdx ].z += d_inf4[ tdx ].z;
        d_inf4[ tdx ].w += d_inf4[ tdx ].w;
    };
};

int main( void )
{
    int gpuCount = 0;
    cudaGetDeviceCount( &gpuCount );

    for ( size_t gpuNo = 0; gpuNo < gpuCount; gpuNo++ )
    {
        cudaSetDevice( gpuNo );
        cudaDeviceProp gpuProperties; cudaGetDeviceProperties( &gpuProperties, gpuNo ); cout << endl << gpuProperties.name << ": " << endl;
        initGPUMem();
        
        for( i = 0; i < nArrays; i++ )
        {    
            auto f1 = chrono::high_resolution_clock::now();
                makeFloat2<<< nBlocks, nThreads >>>( d_arr[ i ] );
                float2_Access<<< ( N / 2 ) / nThreads, nThreads >>>();
                cudaDeviceSynchronize();
            auto f2 = chrono::high_resolution_clock::now();
            cout << "GPU split float2_Access took <chrono> : "
                << chrono::duration_cast< chrono::nanoseconds >( f2 - f1 ).count()
                << " [ns]\n";   
            auto f3 = chrono::high_resolution_clock::now();
                makeFloat3<<< nBlocks, nThreads >>>( d_arr[ i ] );
                float3_Access<<< ( N / 3 ) / nThreads, nThreads >>>();        
                cudaDeviceSynchronize();
            auto f4 = chrono::high_resolution_clock::now();
            cout << "GPU split float3_Access took <chrono> : "
                << chrono::duration_cast< chrono::nanoseconds >( f4 - f3 ).count()
                << " [ns]\n";
            auto f5 = chrono::high_resolution_clock::now();
                makeFloat4<<< nBlocks, nThreads >>>( d_arr[ i ] );     
                float4_Access<<< ( N / 4 ) / nThreads, nThreads >>>();         
                cudaDeviceSynchronize();
            auto f6 = chrono::high_resolution_clock::now();
            cout << "GPU split float4_Access took <chrono> : "
                << chrono::duration_cast< chrono::nanoseconds >( f6 - f5 ).count()
                << " [ns]\n";   
            auto f7 = chrono::high_resolution_clock::now();
                arrFloat2_Access<<< ( N / 2 ) / nThreads, nThreads >>>( d_f2 );       
                cudaDeviceSynchronize();
            auto f8 = chrono::high_resolution_clock::now();
            cout << "GPU pinned arrFloat2_Access took <chrono> : "
                << chrono::duration_cast< chrono::nanoseconds >( f8 - f7 ).count()
                << " [ns]\n";
            auto f9 = chrono::high_resolution_clock::now();
                arrFloat3_Access<<< ( N / 3 ) / nThreads, nThreads >>>( d_f3 );       
                cudaDeviceSynchronize();
            auto f10 = chrono::high_resolution_clock::now();
            cout << "GPU pinned arrFloat3_Access took <chrono> : "
                << chrono::duration_cast< chrono::nanoseconds >( f10 - f9 ).count()
                << " [ns]\n";
            auto f11 = chrono::high_resolution_clock::now();
                arrFloat4_Access<<< ( N / 4 ) / nThreads, nThreads >>>( d_f4 );       
                cudaDeviceSynchronize();
            auto f12 = chrono::high_resolution_clock::now();
            cout << "GPU pinned arrFloat4_Access took <chrono> : "
                << chrono::duration_cast< chrono::nanoseconds >( f12 - f11 ).count()
                << " [ns]\n";
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
                makeFloat2<<< nBlocks, nThreads >>>( d_arr[ i ] );
                float2_Access<<< ( N / 2 ) / nThreads, nThreads >>>();
                cudaDeviceSynchronize();
            cudaEventRecord( stop1 );
            cudaEventSynchronize( stop1 );
            float milliseconds = 0.0f;
            cudaEventElapsedTime( &milliseconds, start1, stop1 );
            cout << "nBlocks[" << nBlocks << "]; nThreads[" << nThreads << "]; GPU split float2_Access took <cudaEvent> : " 
                << milliseconds * 1000000.0f << "[ns]\n";
            cudaEvent_t start2, stop2;
            cudaEventCreate( &start2 );
            cudaEventCreate( &stop2 );                  
            cudaEventRecord( start2 );
                makeFloat3<<< nBlocks, nThreads >>>( d_arr[ i ] );
                float3_Access<<< ( N / 3 ) / nThreads, nThreads >>>();
                cudaDeviceSynchronize();
            cudaEventRecord( stop2 );
            cudaEventSynchronize( stop2 );
            milliseconds = 0.0f;
            cudaEventElapsedTime( &milliseconds, start2, stop2 );
            cout << "nBlocks[" << nBlocks << "]; nThreads[" << nThreads << "]; GPU split float3_Access took <cudaEvent> : " 
                << milliseconds * 1000000.0f << "[ns]\n";
            cudaEvent_t start3, stop3;
            cudaEventCreate( &start3 );
            cudaEventCreate( &stop3 );                  
            cudaEventRecord( start3 );
                makeFloat4<<< nBlocks, nThreads >>>( d_arr[ i ] );
                float4_Access<<< ( N / 4 ) / nThreads, nThreads >>>();
                cudaDeviceSynchronize();
            cudaEventRecord( stop3 );
            cudaEventSynchronize( stop3 );
            milliseconds = 0.0f;
            cudaEventElapsedTime( &milliseconds, start3, stop3 );
            cout << "nBlocks[" << nBlocks << "]; nThreads[" << nThreads << "]; GPU split float4_Access took <cudaEvent> : " 
                << milliseconds * 1000000.0f << "[ns]\n";
            cudaEvent_t start4, stop4;
            cudaEventCreate( &start4 );
            cudaEventCreate( &stop4 );                  
            cudaEventRecord( start4 );
                arrFloat2_Access<<< ( N / 2 ) / nThreads, nThreads >>>( d_f2 );   
                cudaDeviceSynchronize();
            cudaEventRecord( stop4 );
            cudaEventSynchronize( stop4 );
            milliseconds = 0.0f;
            cudaEventElapsedTime( &milliseconds, start4, stop4 );
            cout << "nBlocks[" << ( N / 2 ) / nThreads << "]; nThreads[" << nThreads << "]; GPU pinned arrFloat2_Access took <cudaEvent> : " 
                << milliseconds * 1000000.0f << "[ns]\n";
            cudaEvent_t start5, stop5;
            cudaEventCreate( &start5 );
            cudaEventCreate( &stop5 );                  
            cudaEventRecord( start5 );
                arrFloat3_Access<<< ( N / 3 ) / nThreads, nThreads >>>( d_f3 );   
                cudaDeviceSynchronize();
            cudaEventRecord( stop5 );
            cudaEventSynchronize( stop5 );
            milliseconds = 0.0f;
            cudaEventElapsedTime( &milliseconds, start5, stop5 );
            cout << "nBlocks[" << ( N / 3 ) / nThreads << "]; nThreads[" << nThreads << "]; GPU pinned arrFloat3_Access took <cudaEvent> : " 
                << milliseconds * 1000000.0f << "[ns]\n";
            cudaEvent_t start6, stop6;
            cudaEventCreate( &start6 );
            cudaEventCreate( &stop6 );                  
            cudaEventRecord( start6 );
                arrFloat4_Access<<< ( N / 4 ) / nThreads, nThreads >>>(d_f4);
                cudaDeviceSynchronize();
            cudaEventRecord( stop6 );
            cudaEventSynchronize( stop6 );
            milliseconds = 0.0f;
            cudaEventElapsedTime( &milliseconds, start6, stop6 );
            cout << "nBlocks[" << ( N / 4 ) / nThreads << "]; nThreads[" << nThreads << "]; GPU pinned arrFloat4_Access took <cudaEvent> : " 
                << milliseconds * 1000000.0f << "[ns]\n";                
            cudaEvent_t start7, stop7;
            cudaEventCreate( &start7 );
            cudaEventCreate( &stop7 );                  
            cudaEventRecord( start7 );
                singleThreadAccess<<< 1, 1 >>>( d_arr[ i ] );               
                cudaDeviceSynchronize();
            cudaEventRecord( stop7 );
            cudaEventSynchronize( stop7 );
            milliseconds = 0.0f;
            cudaEventElapsedTime( &milliseconds, start7, stop7 );
            cout << "single thread GPU accesses took <cudaEvent> : " << milliseconds * 1000000.0f << "[ns]\n";                
            cudaEvent_t start8, stop8;
            cudaEventCreate( &start8 );
            cudaEventCreate( &stop8 );                  
            cudaEventRecord( start8 );
                medianThreadAccess<<< nBlocks, nThreads >>>( d_arr[ i ] ); 
                cudaDeviceSynchronize();
            cudaEventRecord( stop8 );
            cudaEventSynchronize( stop8 );
            milliseconds = 0.0f;
            cudaEventElapsedTime( &milliseconds, start8, stop8 );
            cout << "nBlocks[" << nBlocks << "]; nThreads[" << nThreads << "]; GPU accesses took <cudaEvent> : " 
                << milliseconds * 1000000.0f << "[ns]\n";
            nop<<< nBlocks, nThreads >>>( N );
    //============================================================================                
            cudaMemcpy( h_result[ i ], d_arr[ i ], NBytes_f32, D2H );//cudaMemcpyFromSymbol( h_result[ i ], d_arr2, NBytes_f32, H2D );
        };
        for ( ind = 0; ind < 3; ind++ )
            cout << "   h_result[" << ind << "]: " << h_result[ 0 ][ ind ] << endl;
        
        freeGPUMem();
    }; //end of gpuCount
    
    return 0;
}

