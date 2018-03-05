//STL
#include <iostream>
#include <vector>
#include <time.h>
#include <algorithm>

using std::cout; using std::endl; using namespace std;

unsigned i;
const unsigned N = 2048 * 4, bigN = 1000000;
unsigned gpuThr = 512;
unsigned gpuBl = N / gpuThr;
std::vector < float > inputVec( N );
void hostCalculateDCTPSNR( vector < float > &vec, float & vecMedian );
//===========================	gpu ===========================
__device__      float       d_x[ N ], d_Xfp32[ N ], d_ix[ N ], d_rms[ N ];
__constant__    unsigned    d_N[ 1 ];
__constant__    float       d_median[ 1 ], d_max[ 1 ];
__device__      float       d_inOut[ bigN ];
__device__      float       d_inOutCopy[ bigN ];

__global__ void dummyCopy()
{
    unsigned ind = blockIdx.x * blockDim.x + threadIdx.x;
    d_inOutCopy[ ind ] = d_inOut[ ind ];
}

__global__ void psnr()
{
    double acc = 0.0f;
    for ( unsigned i = 0; i < d_N[ 0 ]; i++ )
        acc += d_rms[ i ];
    acc /= float( d_N[ 0 ] );
    printf( "GPU PSNR: %f[dB]\n ", 10.0f * log10f( ( d_max[ 0 ] * d_max[ 0 ] ) / ( acc ) ) );
}

__global__ void rms()
{
    unsigned ind = blockIdx.x * blockDim.x + threadIdx.x;
    float x1 = d_x[ ind ] + d_median[ 0 ];
    float x2 = d_ix[ ind ];
    d_rms[ ind ] = ( x1 - x2 ) * ( x1 - x2 );
}

__global__ void printKernel()
{
    printf( "======= GPU SIDE: =========\n" );
    unsigned resNo = 3;
    for ( unsigned i = 0; i < resNo; i++ )
        printf( "d_x[%i]: %4f\n", i, d_x[ i ] + d_median[ 0 ] );
/*
	for ( unsigned i = 0; i < resNo; i++ )
        printf( "d_xNorm[%i]: %.4f\n", i, d_x[ i ] );
    for ( unsigned i = 0; i < resNo; i++ )
        printf( "d_Xfp32[%i]: %.4f\n", i, d_Xfp32[ i ] );
*/
	for ( unsigned i = 0; i < resNo; i++ )
        printf( "d_ix[%i]: %.4f\n", i, d_ix[ i ] );
    for ( unsigned i = d_N[ 0 ] - 1; i > d_N[ 0 ] - 4; i-- )
        printf( "d_x[%i]: %.4f\n", i, d_x[ i ] + d_median[ 0 ] );
    for ( unsigned i = d_N[ 0 ] - 1; i > d_N[ 0 ] - 4; i-- )
        printf( "d_ix[%i]: %.4f\n", i, d_ix[ i ] );
}

__global__ void idctKernelFloat()
{
    unsigned ind = blockIdx.x * blockDim.x + threadIdx.x;
    float constVal = ( float( ind ) + 0.5f ) * 3.14159265f / float( d_N[ 0 ] );
    float sqrConst = sqrtf( 2.0f / float( d_N[ 0 ] ) );
    float tmpX = sqrtf( 1.0f / float( d_N[ 0 ] ) ) * d_Xfp32[ 0 ];
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
    float constVal = float( ind ) * 3.14159265f / float( d_N[ 0 ] );
    float sqrConst = sqrtf( 2.0f / float( d_N[ 0 ] ) );
    float tmpX = 0.0f, accDC = 0.0f, tmpx = 0.0f;
    for ( unsigned i = 0; i < N; i++ )   
    {
        tmpx = d_x[ i ];
        tmpX += sqrConst * tmpx * __cosf( constVal * ( float( i ) + 0.5f ) );
        accDC += tmpx;
    }
    d_Xfp32[ ind ] = tmpX;
    d_Xfp32[ 0 ] = accDC / sqrtf( float( d_N[ 0 ] ) );
}

//median extraction from input vector <float> for float better calculations precision
__global__ void dataMedianPreprocess()  
{
    unsigned ind = blockIdx.x * blockDim.x + threadIdx.x;
    d_x[ ind ] -= d_median[ 0 ];
}

int main( int argc, char* argv[] )
{//memory copying
    vector < float > h_vecIn;
    for ( i = 0; i < bigN; i++ )
        h_vecIn.push_back( rand() % 100 * 0.01f * i );
    cudaMemcpyToSymbol( d_inOut, &h_vecIn[ 0 ], sizeof( float ) * bigN );
    vector < float > h_vecOut( bigN, 0.0f );
    cudaMemcpyFromSymbol( &h_vecOut[ 0 ], d_inOut, sizeof( float ) * bigN );
    double acc = 0.0f; float x1 = 0.0f, x2 = 0.0f;
    for ( i = 0; i < bigN; i++ )
    {
        x1 = h_vecIn[ i ];
        x2 = h_vecOut[ i ];
        acc += ( x1 - x2 ) * ( x1 - x2 );
    }
    acc /= double( bigN );
    float maxEl = *std::max_element( h_vecIn.begin(), h_vecIn.end() );
    printf( "psnr raw HOST2GPU copy: %f[dB]\n", 10.0f * log10( maxEl * maxEl / acc ) );
    
    dummyCopy<<< bigN / 500, 500 >>>();
    cudaMemcpyFromSymbol( &h_vecOut[ 0 ], d_inOutCopy, sizeof( float ) * bigN );
    acc = 0.0f; x1 = 0.0f; x2 = 0.0f;
    for ( i = 0; i < bigN; i++ )
    {
        x1 = h_vecIn[ i ];
        x2 = h_vecOut[ i ];
        acc += ( x1 - x2 ) * ( x1 - x2 );
    }
    acc /= double( bigN );
    maxEl = *std::max_element( h_vecIn.begin(), h_vecIn.end() );
    printf( "psnr raw GPU2GPU copy: %f[dB]\n", 10.0f * log10( maxEl * maxEl / acc ) );
    cudaFree( d_inOut ); cudaFree( d_inOutCopy );

//gpu DCT from definition accuracuy
    for(i=0;i<(unsigned)inputVec.size();i++)inputVec[i]=rand()%100*0.001f*i; 
    inputVec[ 3 ] = 0.05f;
    vector < float > sortVec( inputVec ); sort( sortVec.begin(), sortVec.end() );
    float vecMedian = sortVec[ sortVec.size() / 2 ];
	cudaMemcpyToSymbol( d_x, &inputVec[ 0 ], sizeof( float ) * ( unsigned )inputVec.size() );
    cudaMemcpyToSymbol( d_N, &N, sizeof( unsigned ) );
    cudaMemcpyToSymbol( d_median, &vecMedian, sizeof( float ) );
    cudaMemcpyToSymbol( d_max, &sortVec[ sortVec.size() - 1 ], sizeof( float ) );
    dataMedianPreprocess<<< gpuBl, gpuThr >>>();
    
    clock_t t = clock();
    dctKernelFloat<<< gpuBl, gpuThr >>>();
    cudaDeviceSynchronize();
    cout << "CPU clocks GPU dct float accumulator: " << double( clock() - t ) << endl;
    
    t = clock();
    idctKernelFloat<<< gpuBl, gpuThr >>>();
    cudaDeviceSynchronize();
    cout << "CPU clocks GPU idct float accumulator: " << double( clock() - t ) << endl;
    printKernel<<< 1, 1 >>>();
    rms<<< gpuBl, gpuThr >>>();
    psnr<<< 1, 1 >>>();

//host DCT from definition accuracy
    hostCalculateDCTPSNR( inputVec, vecMedian );
    
	cudaFree( d_x );
    cudaFree( d_ix );
    cudaFree( d_median );
    cudaFree( d_rms );
    cudaFree( d_max );
    cudaFree( d_Xfp32 );
    cudaFree( d_N );
	cudaDeviceSynchronize();
	cudaDeviceReset();
    cout << endl << "PSNR - higher = better" << endl;
    return 0;
}

void hostCalculateDCTPSNR( vector < float > &vec, float & vecMedian )
{
    clock_t t;
    unsigned vecSize = ( unsigned )vec.size();
    for ( i = 0; i < vecSize; i++ )
        vec[ i ] -= vecMedian;
    vector < float > vecDCT( vecSize );
    vector < float > ix( vecSize );
    
    t = clock();
    float dc = 0.0f;
    for ( i = 0; i < vecSize; i++ )
        dc += vec[ i ];
    dc /= sqrt( vecSize );
    vecDCT[ 0 ] = dc;
    float acDCT = 0.0f, cons = sqrt( 2.0f / vecSize );
    float pi = 3.14159265f;
    for ( unsigned k = 1; k < vecSize; k++ )
    {
        acDCT = 0.0f;
        for ( i = 0; i < vecSize; i++ )
            acDCT += vec[ i ] * cos( pi * k * ( 2 * i + 1 ) / ( 2 * vecSize ) );
        vecDCT[ k ] = cons * acDCT;
    }
    cout << "CPU clocks HOST dct float accumulator: " << double( clock() - t ) << endl;

    t = clock();
    float dcCons = ( 1.0f / sqrt( vecSize ) ) * vecDCT[ 0 ];
    for ( i = 0; i < vecSize; i++ )
    {
        acDCT = 0.0f;
        for ( unsigned k = 1; k < vecSize; k++ )
            acDCT += vecDCT[ k ] * cos( pi * k * ( 2 * i + 1 ) / ( 2 * vecSize ) );
        ix[ i ] = dcCons + cons * acDCT + vecMedian;    //results median addition
    }
    cout << "CPU clocks HOST idct float accumulator: " << double( clock() - t ) << endl;
    
    for ( i = 0; i < vecSize; i++ )
        vec[ i ] += vecMedian;
    
    cout << endl << "======= HOST SIDE: =========" << endl;
    for ( i = 0; i < 3; i++ )
        cout << "h_x[" << i << "]: " << vec[ i ] << endl;
    for ( i = 0; i < 3; i++ )
        cout << "h_ix[" << i << "]: " << ix[ i ] << endl;
    for ( i = vecSize - 1; i > vecSize - 4; i-- )
        cout << "h_x[" << i << "]: " << vec[ i ] << endl;
    for ( i = vecSize - 1; i > vecSize - 4; i-- )
        cout << "h_ix[" << i << "]: " << ix[ i ] << endl;

    double mse = 0.0f;
    for ( i = 0; i < vecSize; i++ )
        mse += ( vec[ i ] - ix[ i ] ) * ( vec[ i ] - ix[ i ] );
    mse /= vecSize;
    double maxEl = *std::max_element( vec.begin(), vec.end() );
    double psnr = 10.0f * log10( maxEl * maxEl / mse ); 
    cout << "HOST PSNR: " << psnr << "[dB]" << endl << endl;
}
//P.S. PSNR( x1[], x2[] ) = +InfdB for identical inputs x1[] and x2[]; PSNR = 0dB for x1[] != x2[]; higher = better accuracy to true/real value
//P.P.S for range [-1; +1] float datatype has biggest mantissa precision 
