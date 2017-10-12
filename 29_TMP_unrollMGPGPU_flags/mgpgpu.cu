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
#define cudaMemcpyHostToDevice H2D;          //useful alias
#define cudaMemcpyDeviceToHost D2H;
__global__ void trivialLoop()
{
    uint a = 0;
    for ( uint32_t i = 0; i < N; i++ )
        a = i;
}

__global__ void unrollTrivialLoop()
{
    uint a = 0;
    #pragma unroll UNROLLING                //briliant feature
    for ( uint32_t i = 0; i < N; i++ )
        a = i;
};

int main( void )
{
    cudaGetDeviceCount( &gpuCount );
    //HOST
    float *h_arr[ gpuCount ];           //     float **h_arr = ( float** )malloc( sizeof( float * ) * gpuCount ); //alternatively
    uint perDevN = 1E3 / gpuCount;
    //DEVICE
    cudaStream_t stream[ gpuCount ];
    float *d_arr[ gpuCount ];           //     float **d_arr = ( float** )malloc( sizeof( float * ) * gpuCount ); //alternatively
    
    //alocate & initialize H,D memories
    for ( i = 0; i < gpuCount; i++ )
    {
        //HOST 
        cudaMallocHost( ( void** ) &h_arr[ i ], perDevN );
        initalizeHost( h_arr[ i ], perDevN );
        //DEVICE
        cudaSetDevice( i );
        cudaMalloc( ( void** ) &d_arr[ i ], perDevN );
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
        printf( "loop elapsed: %d \n", elapsed );
        
        t1 = chrono::high_resolution_clock::now();
        unrollTrivialLoop<<< 1, 1 >>>();
        nop<<< 1, 1 >>>();
        t2 = chrono::high_resolution_clock::now();
        elapsed = chrono::duration_cast< chrono::nanoseconds >( t2 - t1 ).count();  
        printf( "unrolled loop elapsed: %d \n", elapsed );
        
        
        
        
        
        //cudaMemcpyAsync( d_arr[i], h_arr[i], H2D, stream[i] );
        //kernel<<< nT, nB, stream[i] >>>( d_arr, perDevN );
        /*    uint tid = threadIdx.x;
              uint idx = blockIdx.x * blockDim.x + threadIdx.x;
              if( idx < perDevN ) :...
        */
        //cudaMemcpyAsync( gpuRef[i], d_arr, D2H stream[i] );
        
        
        
        
        
        
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
        ip[ i ] = 0.0f;
};

//Post Scriptum: In my professional opinion, coprocessors: GTX1080ti is brand-new and off-the-shell optimal; GTX770 is used optimal - I've heard about R9Nano and HD5770 ( GFLOPS/USD; GFLOPS/W; QualityWithBandwidthAndMemSize/Price; ); 
