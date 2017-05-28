//STL
#include <iostream>
#include <vector>
#include <thread>
#include <time.h>

using namespace std;

////////////////////////////////////////// HOST //////////////////////////////////////////////////////
int N = 20000;  //GPU calculations are effective for N > 8k
vector < vector < float > > firstMatrix( N );
vector < vector < float > > secondVectorMatrix( N );
vector < vector < float > > resultsHostMatrix( N );
vector < vector< float > > resultsGPUMatrix( N );
vector < float > firstVector( N, 3.14 );
vector < float > secondVector( N, 2.72 );
vector < float > resultsHost( N );
vector < float > wynikGPU( N, 0 ); 
////////////////////////////////////////// GPU ////////////////////////////////////////////////////////

    float *dev_a1 = 0; float *dev_b1 = 0; float *dev_c1 = 0;
    int typeSize = N * sizeof( float );

    __global__ void add( float *a, float *b, float *c, int N )                 //GPU
    {
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        while ( tid < N )
        {
            c[ tid ] = a[ tid ] + b[ tid ];
            tid += blockDim.x * gridDim.x;
        }
    }

    void memInit()
    {
            //GPU memory allocation
            cudaMalloc( ( void** )&dev_a1, typeSize );
            cudaMalloc( ( void** )&dev_b1, typeSize );
            cudaMalloc( ( void** )&dev_c1, typeSize );
    }
    
    void memFree()
    {
            //free GPU objects
            cudaFree( dev_a1 );
            cudaFree( dev_b1 );
            cudaFree( dev_c1 );
    }
    
    void addMatrixKernel( void *iter )
    {
        if ( *( int * )iter < N )
        {
            const int nThreads = 1024;            
            //copy / download data in direction HostToDevice
            cudaMemcpyAsync( dev_a1, &firstMatrix[ *( int * ) iter  ][ 0 ], typeSize, cudaMemcpyHostToDevice );
            cudaMemcpyAsync( dev_b1, &secondVectorMatrix[ *( int * ) iter  ][ 0 ], typeSize, cudaMemcpyHostToDevice );
            //calculate vectors sum, using max. number of possible 1D Threads per Block
            add<<< ( N + nThreads - 1 ) / nThreads, nThreads >>> ( dev_a1, dev_b1, dev_c1, N );
            //copy / upload results data c[] in direction DeviceToHost
            cudaMemcpyAsync( &resultsGPUMatrix[ *( int * ) iter  ][ 0 ], dev_c1, typeSize, cudaMemcpyDeviceToHost );
        }
    }
    
////////////////////////////////////////// MAIN ///////////////////////////////////////////////////////
int main ()
{
    for ( int i = 0; i < N; i++ )				//basic data processing on Host CPU
    {
        firstMatrix[ i ] = firstVector;
        secondVectorMatrix[ i ] = secondVector;
        resultsGPUMatrix[ i ] = wynikGPU;
        resultsHostMatrix[ i ] = resultsHost;
    }
    clock_t t;
    t = clock();

    for ( int j = 0; j < N; j++ )
        for ( int i = 0; i < N; i++ )
            resultsHostMatrix[ j ][ i ] = firstMatrix[ j ][ i ] + secondVectorMatrix[ j ][ i ];    
    cout << "sequential CPU calculations Host time: " << ((float)(clock() - t))/CLOCKS_PER_SEC << "[s] ( g++ )" << endl;

    t = clock();
    memInit();
    vector < thread > gpuAsync3( N );
    for ( int i = 0; i < N; i++ )
    {
        int *iPtr = &( i );   
        gpuAsync3[ i ] = thread( addMatrixKernel, iPtr ); 
        gpuAsync3[ i ].join();                   
    }    
    memFree();
    cout << "Async (single join() + trivial Optimalization) vec<vec<>> GPU time: " << ((float)(clock() - t))/CLOCKS_PER_SEC << "[s]" << endl;
    t = clock();
    
    for ( int i = 0; i < 2; i++ )
    {
        cout << "resultsHostMatrix[ " << i << " ][ 0 ]: " << resultsHostMatrix[ i ][ 0 ] << endl;
        cout << "resultsGPUMatrix[ " << i << " ][ 0 ]: " << resultsGPUMatrix[ i ][ 0 ] << endl;
    }
    
    cudaDeviceReset();
    return 0;
}

