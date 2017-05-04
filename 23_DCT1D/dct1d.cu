//STL
#include <iostream>
#include <string>
#include <vector>
#include <time.h>

using namespace std;

unsigned i;
const unsigned N = 128;
unsigned gpuThr = 32;
unsigned gpuBl = N / gpuThr;
vector < float > inputVec( N );
string letter, subFp; const string sep( "_" );

//===========================	gpu ===========================
__device__      float d_x[ N ], d_Xfp32[ N ], d_ix[ N ];
__device__      double d_Xfp64[ N ]; 	//double size per dimension in comparision to floats array in global memory; for 2D results in quadratic size
__constant__    unsigned d_N[ 1 ];
__global__ void printKernel()
{
    unsigned resNo = 10;
	for ( unsigned i = 0; i < resNo; i++ )
        printf( "x[%i]: %f\n", i, d_x[ i ] );
    for ( unsigned i = 0; i < resNo; i++ )
        printf( "d_Xfp32[%i]: %.6f;   d_Xfp64[%i]: %.6f\n", i, d_Xfp32[ i ], i, d_Xfp64[ i ] );
	for ( unsigned i = 0; i < resNo; i++ )
        printf( "ix[%i]: %f\n", i, d_ix[ i ] );
    double acc = 0.0f;
    for( unsigned i = 0; i < N; i++ )
        acc += ( d_Xfp32[ i ] - d_Xfp64[ i ] ) * ( d_Xfp32[ i ] - d_Xfp64[ i ] );
    acc /= N;
    printf( "mean difference in dct float vs double accumulators: %.6f\n", sqrtf( acc ) );
}

__global__ void idctKernelFloat()
{
    unsigned ind = blockIdx.x * blockDim.x + threadIdx.x;
    float constVal = float( ind ) * 3.14159265f / float( N );
    float sqrConst = sqrtf( 2.0f / float( N ) );
    float tmpX = sqrtf( 1.0f / float( N ) ) * d_Xfp32[ 0 ];
    float accDC = 0.0f, tmpx = 0.0f; 
    for ( unsigned i = 1; i < N; i++ )   
    {
        tmpx = d_Xfp32[ i ];
        tmpX += tmpx * sqrConst * __cosf( constVal * ( float( i ) + 0.5f ) );
        accDC += tmpx;
    }
    d_ix[ ind ] = tmpX;
}

__global__ void dctKernelFloat()
{
    unsigned ind = blockIdx.x * blockDim.x + threadIdx.x;
    float constVal = float( ind ) * 3.14159265f / float( N );
    float sqrConst = sqrtf( 2.0f / float( N ) );
    float tmpX = 0.0f, accDC = 0.0f, tmpx = 0.0f;
    for ( unsigned i = 0; i < N; i++ )   
    {
        tmpx = d_x[ i ];
        tmpX += sqrConst * tmpx * __cosf( constVal * ( float( i ) + 0.5f ) );
        accDC += tmpx;
    }
    d_Xfp32[ ind ] = tmpX;
    d_Xfp32[ 0 ] = accDC / sqrtf( float( N ) );
}

__global__ void dctKernelDouble()
{
    unsigned ind = blockIdx.x * blockDim.x + threadIdx.x;
    float constVal = float( ind ) * 3.14159265f / float( N );
    float sqrConst = sqrtf( 2.0f / float( N ) );
    double tmpX = 0.0f, accDC = 0.0f, tmpx = 0.0f;
    for ( unsigned i = 0; i < N; i++ )   
    {
        tmpx = d_x[ i ];
        tmpX += sqrConst * tmpx * __cosf( constVal * ( float( i ) + 0.5f ) );
        accDC += tmpx;
    }
    d_Xfp64[ ind ] = tmpX;
    d_Xfp64[ 0 ] = accDC / sqrtf( float( N ) );
}

int main( int argc, char* argv[] )
{
    for(i=0;i<(unsigned)inputVec.size();i++)inputVec[i]=0.1f*i;
	cudaMemcpyToSymbol( d_x, &inputVec[ 0 ], sizeof( float ) * ( unsigned )inputVec.size() );
    cudaMemcpyToSymbol( d_N, &N, sizeof( unsigned ) );
    clock_t t = clock();
    dctKernelFloat<<< gpuBl, gpuThr >>>();
    cudaDeviceSynchronize();
    cout << "CPU clocks float accumulator: " << double( clock() - t ) << endl;
    
    t = clock();
    dctKernelDouble<<< gpuBl, gpuThr >>>();
    cudaDeviceSynchronize();
    cout << "CPU clocks double accumulator: " << double( clock() - t ) << endl;
    
    t = clock();
    idctKernelFloat<<< gpuBl, gpuThr >>>();
    cudaDeviceSynchronize();
    cout << "CPU clocks idct float accumulator: " << double( clock() - t ) << endl;
    printKernel<<< 1, 1 >>>();
    
	cudaFree( d_x );
    cudaFree( d_ix );
    cudaFree( d_Xfp32 );
    cudaFree( d_Xfp64 );
    cudaFree( d_N );
	cudaDeviceSynchronize();
	cudaDeviceReset();
    return 0;
}
//P.S. Please note, that streaming data from GPU to RAM is costly in both directions - keep computations in GPU.
