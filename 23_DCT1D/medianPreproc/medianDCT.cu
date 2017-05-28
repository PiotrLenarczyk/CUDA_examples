//STL
#include <iostream>
#include <vector>
#include <time.h>
#include <algorithm>

using std::cout; using std::endl; using namespace std;

unsigned i;
const unsigned N = 2048;
unsigned gpuThr = 256;
unsigned gpuBl = N / gpuThr;
std::vector < float > inputVec( N );

//===========================	gpu ===========================
__device__      float       d_x[ N ], d_Xfp32[ N ], d_ix[ N ];
__constant__    unsigned    d_N[ 1 ];
__constant__    float       d_median[ 1 ];
__global__ void printKernel()
{
    unsigned resNo = 10;
	for ( unsigned i = 0; i < resNo; i++ )
        printf( "d_x[%i] + d_median[ 0 ]: %f\n", i, d_x[ i ] + d_median[ 0 ] );
    for ( unsigned i = 0; i < resNo; i++ )
        printf( "d_Xfp32[%i]: %.6f\n", i, d_Xfp32[ i ] );
	for ( unsigned i = 0; i < resNo; i++ )
        printf( "d_ix[%i]: %f\n", i, d_ix[ i ] );
}

__global__ void idctKernelFloat()
{
    unsigned ind = blockIdx.x * blockDim.x + threadIdx.x;
    float constVal = ( float( ind ) + 0.5f ) * 3.14159265f / float( N );
    float sqrConst = sqrtf( 2.0f / float( N ) );
    float tmpX = sqrtf( 1.0f / float( N ) ) * d_Xfp32[ 0 ];
    float accDC = 0.0f, tmpx = 0.0f; 
    for ( unsigned k = 1; k < N; k++ )   
    {
        tmpx = d_Xfp32[ k ];
        tmpX += tmpx * sqrConst * __cosf( constVal * ( float( k ) ) );
        accDC += tmpx;
    }
    d_ix[ ind ] = tmpX + d_median[ 0 ];
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

__global__ void dataMedianPreprocess()
{
    unsigned ind = blockIdx.x * blockDim.x + threadIdx.x;
    d_x[ ind ] -= d_median[ 0 ];
}

int main( int argc, char* argv[] )
{
    for(i=0;i<(unsigned)inputVec.size();i++)inputVec[i]=0.1f*i; 
    inputVec[ 3 ] = 0.05f;
    vector < float > sortVec( inputVec ); sort( sortVec.begin(), sortVec.end() );
    float vecMedian = sortVec[ sortVec.size() / 2 ];
    cout << "vector median: " << vecMedian << endl;
	cudaMemcpyToSymbol( d_x, &inputVec[ 0 ], sizeof( float ) * ( unsigned )inputVec.size() );
    cudaMemcpyToSymbol( d_N, &N, sizeof( unsigned ) );
    cudaMemcpyToSymbol( d_median, &vecMedian, sizeof( float ) );
    dataMedianPreprocess<<< gpuBl, gpuThr >>>();
    
    clock_t t = clock();
    dctKernelFloat<<< gpuBl, gpuThr >>>();
    cudaDeviceSynchronize();
    cout << "CPU clocks float accumulator: " << double( clock() - t ) << endl;
    
    t = clock();
    idctKernelFloat<<< gpuBl, gpuThr >>>();
    cudaDeviceSynchronize();
    cout << "CPU clocks idct float accumulator: " << double( clock() - t ) << endl;
    printKernel<<< 1, 1 >>>();
    
	cudaFree( d_x );
    cudaFree( d_ix );
    cudaFree( d_median );
    cudaFree( d_Xfp32 );
    cudaFree( d_N );
	cudaDeviceSynchronize();
	cudaDeviceReset();
    return 0;
}
//P.S. Slightly better results for extracting median value from input data.
