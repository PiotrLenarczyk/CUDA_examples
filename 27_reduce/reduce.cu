/*on the basis of "Professional CUDA C Programming" et al. John Cheng */
//STL
#include <iostream>
#include <vector>
#include <chrono>

using namespace std;

//Host
const unsigned N = 1024 * 12; //send most of host data to GPU
unsigned i = 0;
vector< float > vec( N, 2.0f ); //sample vector;

//GPU
const unsigned nThreads = 512;
const unsigned nBlocks = ( N / nThreads ) + 1;
const unsigned unRoll = 8;
__device__ float d_i[ N ]; float *d_iPtr;
__device__ float d_o[ N ]; float *d_oPtr;

void freeGpu()
{
    cudaFree( d_i );
    cudaFree( d_o );
    cudaDeviceReset();
}
__global__ void reduction( float *d_in, float *d_out )
{
    unsigned tid = threadIdx.x;
    unsigned idx = blockIdx.x * blockDim.x * unRoll + threadIdx.x;

    //kernel-passed input data symbol pointer to ptr belonging to this block
    float *bl_d_iPtr = d_in + blockIdx.x * blockDim.x * unRoll;

    float a0, a1, a2, a3, a4, a5, a6, a7;
    if ( idx + ( unRoll - 1 ) * blockDim.x < N )
    {
        a0 = d_in[ idx ];
        a1 = d_in[ idx + blockDim.x ];
        a2 = d_in[ idx + 2 * blockDim.x ];
        a3 = d_in[ idx + 3 * blockDim.x ];
        a4 = d_in[ idx + 4 * blockDim.x ];
        a5 = d_in[ idx + 5 * blockDim.x ];
        a6 = d_in[ idx + 6 * blockDim.x ];
        a7 = d_in[ idx + 7 * blockDim.x ];
        d_in[idx] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    }
    __syncthreads();

    for ( unsigned inWarp = blockDim.x / 2; inWarp > 32; inWarp >>= 1 )
    {
        if ( tid < inWarp )
            bl_d_iPtr[ tid ] += bl_d_iPtr[ tid + inWarp ];
        __syncthreads();
    }

    if ( tid < 32 )
    {
        volatile float *globMem = bl_d_iPtr;
        globMem[ tid ] += globMem[ tid + 32 ];
        globMem[ tid ] += globMem[ tid + 16 ];
        globMem[ tid ] += globMem[ tid + 8 ];
        globMem[ tid ] += globMem[ tid + 4 ];
        globMem[ tid ] += globMem[ tid + 2 ];
        globMem[ tid ] += globMem[ tid + 1 ];
    }

    if ( tid == 0 ) 
        d_out[ blockIdx.x ] = bl_d_iPtr[ 0 ];
}

__global__ void syncKernel(){}
unsigned aproximateSyncKernel()
{
    auto t1 = std::chrono::high_resolution_clock::now();
    syncKernel<<< 1, 1 >>>();
    auto t2 = std::chrono::high_resolution_clock::now();
    return unsigned( std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() );
}

int main( void )
{   cudaDeviceSynchronize();

    cudaGetSymbolAddress( ( void ** )&d_iPtr, d_i );
    cudaGetSymbolAddress( ( void ** )&d_oPtr, d_o );
    cudaMemcpyToSymbolAsync( d_i, &vec[ 0 ], sizeof( float ) * N );
    unsigned tDif = aproximateSyncKernel();
    auto t1 = std::chrono::high_resolution_clock::now();
        reduction<<< nBlocks, nThreads >>>( d_iPtr, d_oPtr );
        syncKernel<<< 1, 1 >>>();
    auto t2 = std::chrono::high_resolution_clock::now(); 
    tDif -= std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
    std::cout << "reduction of float array[" << N << "] took " << tDif << " [ns]\n";
    
    cudaMemcpyFromSymbolAsync( &vec[ 0 ], d_o, sizeof( float ) * N );
    double gpu_sum = 0;
    for ( i = 0; i < nBlocks / 8; i++ ) 
        gpu_sum += vec[ i ];
    if ( gpu_sum !=  double( 2 * N ) ) { cerr << "Reduction failed!\n"; return -1; }
    cout << "sum: " << gpu_sum << endl; //reduction sum should equal to 2 * N

    freeGpu();
    return 0;
}
//P.S. quite tricky reduction.
