#include <iostream>
#include <chrono>

using namespace std;

//CPU
typedef unsigned int uint;
uint i = 0;
int gpuCount = 0;
void initalizeHost( float *ip, uint size );

//GPU
cudaDeviceProp gpuProperties;
const uint N = 1E7;
const uint nThreads = 512;
const uint nBlocks = ( N / nThreads ) + 1;
const uint UNROLLING = 16;                   //check [ 8 16 32 64 ]; I would guess sixteen times unrolling;
__global__ void nop(){}; 

__global__ void trivialLoop()
{
    uint a = 0;
    for ( uint32_t i = 0; i < N; i++ )
    {
        a = i;
        a++;
    }
};

__global__ void unrollTrivialLoop()
{
    uint a = 0;
    #pragma unroll UNROLLING                //briliant feature
    for ( uint32_t i = 0; i < N; i++ )
    {
        a = i;
        a++;
    }
};

__global__ void initializeDevice( float *d_in, const uint streamSize )
{
    uint tid = threadIdx.x;
    uint idx = blockIdx.x * blockDim.x + tid;
    if( idx < streamSize )
    {
        d_in[ idx ] *= 0.9f;
    }
};

__global__ void loop( float *d_in, const uint streamNo, const uint streamSize )
{
    size_t ii = 0;
    for ( ii = 0; ii < streamSize; ii++ )
    {
        d_in[ ii ] +=  float( ii );
    }
};

__global__ void unrollLoop( float *d_in, const uint streamNo, const uint streamSize )
{
    size_t ii = 0;
    #pragma unroll UNROLLING            //adjustable mainly to register-only, high performance computations - described at unrollTrivialLoop device kernel
    for ( ii = 0; ii < streamSize; ii++ )
    {
        d_in[ ii ] +=  float( ii );
    }
};

int main( void )
{
    cudaGetDeviceCount( &gpuCount );
    //HOST
    float *h_arr[ gpuCount ];           //     float **h_arr = ( float** )malloc( sizeof( float * ) * gpuCount ); //alternatively
    uint perDevN = N / gpuCount;
    uint perDevNBytes = sizeof( float ) * perDevN ;
    //DEVICE
    cudaStream_t stream[ gpuCount ];
    float *d_arr[ gpuCount ];           //     float **d_arr = ( float** )malloc( sizeof( float * ) * gpuCount ); //alternatively
    
    //alocate & initialize H,D memories
    for ( i = 0; i < gpuCount; i++ )
    {
        //HOST 
        cudaMallocHost( ( void** ) &h_arr[ i ], perDevNBytes );
        initalizeHost( h_arr[ i ], perDevN );
        //DEVICE
        cudaSetDevice( i );
        cudaMalloc( ( void** ) &d_arr[ i ], perDevNBytes );
        cudaStreamCreate( &stream[ i ] );
    }
    
    //DEVICE computations
    for ( i = 0; i < gpuCount; i++ )
    {
        cudaSetDevice( i );
        cudaGetDeviceProperties( &gpuProperties, i );
        cout << endl << gpuProperties.name << ": " << endl;
        auto t1 = chrono::high_resolution_clock::now();
        trivialLoop<<< 1, 1 >>>();
        nop<<< 1, 1 >>>();
        auto t2 = chrono::high_resolution_clock::now();
        uint elapsed = uint( chrono::duration_cast< chrono::nanoseconds >( t2 - t1 ).count() ); 
        printf( "trivial loop elapsed: %d \n", elapsed );
        
        t1 = chrono::high_resolution_clock::now();
        unrollTrivialLoop<<< 1, 1 >>>();
        nop<<< 1, 1 >>>();
        t2 = chrono::high_resolution_clock::now();
        elapsed = chrono::duration_cast< chrono::nanoseconds >( t2 - t1 ).count();  
        printf( "trivial unrolled loop elapsed: %d \n", elapsed );
        
        
        cudaMemcpyAsync( d_arr[ i ], h_arr[ i ], perDevNBytes, cudaMemcpyHostToDevice, stream[ i ] );
        initializeDevice<<< nBlocks, nThreads, 0, stream[ i ] >>>( d_arr[ i ], perDevN );
        nop<<< 1, 1 >>>();
        t1 = chrono::high_resolution_clock::now();
        loop<<< 1, 1, 0, stream[ i ] >>>( d_arr[ i ], i, perDevN );
        nop<<< 1, 1 >>>();
        t2 = chrono::high_resolution_clock::now();
        elapsed = chrono::duration_cast< chrono::nanoseconds >( t2 - t1 ).count();  
        printf( "loop elapsed: %d \n", elapsed );
        
        t1 = chrono::high_resolution_clock::now();
        unrollLoop<<< 1, 1, 0, stream[ i ] >>>( d_arr[ i ], i, perDevN );
        nop<<< 1, 1 >>>();
        t2 = chrono::high_resolution_clock::now();
        elapsed = chrono::duration_cast< chrono::nanoseconds >( t2 - t1 ).count();  
        printf( "unrolled loop elapsed: %d \n", elapsed );
    }
    
    //free memories
    for ( i = 0; i < gpuCount; i++ )
    {
        //HOST 
        cudaFreeHost( h_arr[ i ] );
        //DEVICE
        cudaSetDevice( i );
        cudaFree( d_arr[ i ] );
        cudaStreamDestroy( stream[ i ] );
    }

    cudaDeviceReset();
    return 0;
}

void initalizeHost( float *ip, uint size )
{
    for ( size_t i = 0; i < size; i++ )
        ip[ i ] = 1.2f;
};

//Post Scriptum: In my professional opinion, coprocessors: GTX1080ti is brand-new and off-the-shell optimal; GTX770 is used optimal - I've heard about R9Nano and HD5770 ( GFLOPS/USD; GFLOPS/W; QualityWithBandwidthAndMemSize/Price; ); 
//Post Post Scriptum: I do strongly recommend profiling with NVidia's NVPROF profiling tool, instead of CPU high-resolution timer.
