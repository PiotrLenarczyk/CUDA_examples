//STL
#include <cstdlib>                  //atol
#include <stdlib.h>
#include <iostream>                 //printf, cout, endl
//THRUST
#include <thrust/device_vector.h>   //omit host vectors and costly data transfers
#include <thrust/for_each.h>        //normalize with size inversed FFT
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <device_ptr.h>
//cuFFT
#include <cufft.h>

using namespace thrust;             //note that default is THRUST!!!
using std::cout; using std::endl;   //STL is for data viewing

unsigned N( 0 ); __constant__ unsigned d_N[ 1 ];
__constant__ float d_LUTspat[ 1920 ]; 
__constant__ float d_sqrConst[ 1 ];
const float h_PI = 3.1415926535897932384626;
/*
__global__ void grandChildKernel( float* spatInChild, float* childSpecCoeffNo, float* dctTmpChild )    //up 65536 blocks for single thread
{
    unsigned blockId = blockIdx.x;
    dctTmpChild[ blockId ] = spatInChild[ blockId ] * __cosf( *childSpecCoeffNo * d_LUTspat[ blockId ]   );
}

__global__ void parentKernel( float* specCoeffNo, float* spatIn, float* dctTmp, float* specOut )       
{
    for ( *specCoeffNo = 1; *specCoeffNo < 1920; *specCoeffNo += 1 )     //note child kernel  blockIDx = *specCoeffNo
    {
        grandChildKernel<<< 1920, 1 >>>( spatIn, specCoeffNo, dctTmp );
        cudaDeviceSynchronize();
        specOut[ unsigned( *specCoeffNo ) ] = reduce( seq, dctTmp, dctTmp + d_N[ 0 ] ) *  d_sqrConst[ 0 ]; //note float lossy quantizer noise as trade-off: speed/accuracy
//         if ( unsigned( *specCoeffNo ) < 10 )
//             printf( "DCT[%i]: %f\n", unsigned( *specCoeffNo ), specOut[ unsigned( *specCoeffNo ) ] ); 
    }
}
*/

__global__ void grandChildKernel( float* spatInChild, float* childSpecCoeffNo, float &dctTmpChild )    //up 65536 blocks for single thread
{
    unsigned blockId = blockIdx.x;
    dctTmpChild[ unsigned( childSpecCoeffNo ) ][ blockId ] = spatInChild[ blockId ] * __cosf( *childSpecCoeffNo * d_LUTspat[ blockId ]   );
}

__device__ float LUTs [ LUTm ][ LUTk ]; //lowers global data transfers
__device__ float dctTmp2D[ 1920 ][ 1920 ]; 

__global__ void parentKernel( float* specCoeffNo, float* spatIn, float* dctTmp2D, float* specOut )       
{
    for ( *specCoeffNo = 1; *specCoeffNo < 1920; *specCoeffNo += 1 )     //note child kernel  blockIDx = *specCoeffNo
    {
        grandChildKernel<<< 1920, 1 >>>( spatIn, specCoeffNo, dctTmp2D[ unsigned( *specCoeffNo ) ] );
        cudaDeviceSynchronize();
//         specOut[ unsigned( *specCoeffNo ) ] = reduce( seq, dctTmp2D[ unsigned( *specCoeffNo ) ], dctTmp2D[ unsigned( *specCoeffNo ) ] + d_N[ 0 ] ) *  d_sqrConst[ 0 ]; //note float lossy quantizer noise as trade-off: speed/accuracy
//         if ( unsigned( *specCoeffNo ) < 10 )
//             printf( "DCT[%i]: %f\n", unsigned( *specCoeffNo ), specOut[ unsigned( *specCoeffNo ) ] ); 
    }
}
 
int main( int argc, char *argv[] )  //DCT 1D is fit to size 1920!!!
{
    clock_t t( clock() );
    N = atoi( argv[ argc - 1 ] ); 
    if ( N == 0 ) 
    {
        cout << "There are no input arguments!" << endl;
        cudaDeviceSynchronize();
        return 0;
    }
    else
        printf( "N = %i\n", N );
    cudaMemcpyToSymbol( d_N, &N, sizeof( float ) );
    
////    THRUST example       h_PI 
    int ROWY = 2;
    int COLX = N;
    device_vector< float > dim1( COLX ); sequence( device, dim1.begin(), dim1.end(), 0.1f );   
    device_vector< float > d_vecD[ ROWY ]; 
    device_vector< float > d_vecD_DCT[ ROWY ];
    for ( unsigned i = 0; i < ROWY; i++ )       //vecD must be init!__global__ void childKernel( float* spatInChild
    {
        d_vecD[ i ] = dim1;
        d_vecD_DCT[ i ] = dim1;
    }
    cout << "d_vecD[ 0 ][ 1 ]: " << d_vecD[ 0 ][ 1 ] << endl;
    cout << "d_vecD_Row[0].size(): " << int( device_vector< float >( d_vecD[ 0 ] ).size() ) << endl;
    
//    //further processing 1D d_vecD_Row[0]
    float h_sqrtConst = sqrt( 2.0f / float( N ) ); //constant device variable sqrt(2/N)
    cudaMemcpyToSymbol( d_sqrConst, &h_sqrtConst, sizeof( float ) );
    
    float h_LUTspat[ N ];   //constant device variable LUT spatial argument of freq. cosf()
    for ( unsigned i = 0; i < N; i++ )
        h_LUTspat[ i ] = ( h_PI / float( N ) ) * ( i + 0.5f );
    cudaMemcpyToSymbol( d_LUTspat, &h_LUTspat, N * sizeof( float ) );
    
////note DYNAMIC PARALLELISM FOR EACH - writing kernel:
//  //DCT1D( d_vecD[ 0 ] ) DC freq.
    d_vecD_DCT[ 0 ][ 0 ] = reduce( d_vecD[0].begin(), d_vecD[0].end() ) / sqrt( float( N ) );
    cout << "d_vecD_DCT[ 0 ][ 0 ]: " << d_vecD_DCT[ 0 ][ 0 ] << endl;
    
//  //DCT1D( d_vecD[ 0 ] ) AC freq.    
//I) g(m)*__cosf(LUTspat*k)
    float *d_vecD_tmpTransfarray;     //shared memory variable candidate -        note FOR_EACH CANDIDATE also P.S. one Malloc for any
    cudaMalloc( ( void** ) &d_vecD_tmpTransfarray, sizeof( float ) * N );
    thrust::copy( d_vecD[ 0 ].begin(), d_vecD[ 0 ].end(), thrust::device_ptr< float >(d_vecD_tmpTransfarray)); //d_vecD_DCT[ 0 ] vector
    float *d_vecD_DCTpriv;     
    cudaMalloc( ( void** ) &d_vecD_DCTpriv, sizeof( float ) * N );
    float *d_tmpSpecCoeffNo;     
    cudaMalloc( ( void** ) &d_tmpSpecCoeffNo, sizeof( double ) * N );
    
    float *d_vecD_DCTarray;     
    cudaMalloc( ( void** ) &d_vecD_DCTarray, sizeof( float ) * N );

    parentKernel<<< 1, 1 >>>( d_tmpSpecCoeffNo, d_vecD_tmpTransfarray, d_vecD_DCTpriv, d_vecD_DCTarray );    //up to 1024 dynPar threads!
    
    
    cudaFree( d_tmpSpecCoeffNo );           //tmp spectrum coefficient number
    cudaFree( d_vecD_DCTpriv );             //DCT tmp var
    cudaFree( d_vecD_DCTarray );            //DCT 1D result
    cudaFree( d_vecD_tmpTransfarray );   //dynPar deallocation
    cudaFree( d_sqrConst );                 //sqrt 2/N free
    cudaFree( d_LUTspat );                  //spatial LUT
    cudaFree( d_N );                        //device size N
    cudaDeviceSynchronize();
    return 0;
}





