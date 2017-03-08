//STL
#include "HOST_GPU.h"

//GLOBALSY GPU
__constant__ int dev_N = N;
const int nThreads = 1024;
const int nBlocks = ( N + nThreads - 1 ) / nThreads;
float *dev_a1 = 0; float *dev_b1 = 0; float *dev_c1 = 0; float *dev_tmp = 0;
float *dev_a1Inv = 0;
int typeSize = N * sizeof( float );

cudaError_t cuda_main( int &iter )
{  
    cout << "N:" << N << endl;
    return cudaGetLastError();
}

__global__ void meanSequential( float *a, float *b, float *c_partial )                 //GPU
{
    float tmpSingleThreadSumA = 0.0f;
    float tmpSingleThreadSumB = 0.0f;
    for ( int i = 0; i < dev_N; i++ )
    {
        tmpSingleThreadSumA += a[ i ];
        tmpSingleThreadSumB += b[ i ];
    }
    c_partial[ 0 ] = tmpSingleThreadSumA / dev_N;
    c_partial[ 1 ] = tmpSingleThreadSumB / dev_N;
}

__global__ void meanSharedReduction( float *a, float *b, float *c_partial )                 //GPU
{
    __shared__ float vecSumA[ nThreads ]; //block partial sums
    __shared__ float vecSumB[ nThreads ]; //block partial sums
    float tmpSingleThreadSumA = 0.0f;
    float tmpSingleThreadSumB = 0.0f;
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    while ( threadID < dev_N )
    {
        tmpSingleThreadSumA += a[ threadID ];
        tmpSingleThreadSumB += b[ threadID ];
        threadID += blockDim.x * gridDim.x;
    }
    vecSumA[ threadIdx.x ] = tmpSingleThreadSumA;
    vecSumB[ threadIdx.x ] = tmpSingleThreadSumB;
    __syncthreads();                                                       //one thread can save to shared memory at once
    int i = blockDim.x / 2;
    while ( i != 0 )
    {
        if ( threadIdx.x < i )
        {
            vecSumA[ threadIdx.x ] += vecSumA[ threadIdx.x + i ];
            vecSumB[ threadIdx.x ] += vecSumB[ threadIdx.x + i ];
        }
        __syncthreads();
        i /= 2;
    }
    if ( threadIdx.x == 0 )
    {
        c_partial[ 0 ] = vecSumA[ 0 ] / dev_N; //block partial sums
        c_partial[ 1 ] = vecSumB[ 0 ] / dev_N; //block partial sums
    }
}

__global__ void inverse( float *a, float *aInv )
{
    int threadID = threadIdx.x + blockIdx.x * blockDim.x;
    while ( threadID < dev_N )
    {
        aInv[ threadID ] = 1 / a[ threadID ];
        threadID += blockDim.x * gridDim.x;
    }
}

void memInit()
{
        //GPU memory allocation
        HANDLE_ERROR( cudaMalloc( ( void** )&dev_a1, typeSize ) );
        HANDLE_ERROR( cudaMalloc( ( void** )&dev_a1Inv, typeSize ) );
        HANDLE_ERROR( cudaMalloc( ( void** )&dev_b1, typeSize ) );
        HANDLE_ERROR( cudaMalloc( ( void** )&dev_c1, 2 * sizeof( float ) ) );
        HANDLE_ERROR( cudaMalloc( ( void** )&dev_tmp, 1 * sizeof( float ) ) );
}

void memFree()
{
        //GPU memory free
        cudaFree( dev_a1 );
        cudaFree( dev_a1Inv );
        cudaFree( dev_b1 );
        cudaFree( dev_c1 );
        cudaFree( dev_tmp );
        cudaDeviceReset();
}

void dodajMatJadro( void *iter )
{
    if ( *( int * )iter < N )
    {
        //copy data  HostToDevice
        HANDLE_ERROR( cudaMemcpyAsync( dev_a1, &firstMat[ *( int * ) iter  ][ 0 ], typeSize, cudaMemcpyHostToDevice ) );
        HANDLE_ERROR( cudaMemcpyAsync( dev_b1, &secondVecMat[ *( int * ) iter  ][ 0 ], typeSize, cudaMemcpyHostToDevice ) );
        meanSharedReduction<<< 1, nThreads >>> ( dev_a1, dev_b1, dev_c1 );               //1 block, nThreads, parallelization = N /( nBlocks * nThreads )
        inverse<<< nBlocks, nThreads >>> ( dev_a1, dev_a1Inv );                         //parallelization = N /( nBlocks * nThreads )
        //copy data c[] DeviceToHost
        HANDLE_ERROR( cudaMemcpyAsync( &resultsGPU[ *( int * ) iter  ][ 0 ], dev_c1, 2 * sizeof( float ), cudaMemcpyDeviceToHost ) );
    }
}
