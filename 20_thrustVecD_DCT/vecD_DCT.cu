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

unsigned N( 0 );
__constant__ float d_LUTspat[ 1920 ]; 
__constant__ float d_sqrConst[ 1 ];
const float h_PI = 3.1415926535897932384626;

__global__ void childKernel( float* spatInChild, float childSpecCoeffNo, float* dctTmpChild )    //up 65536 blocks for single thread
{
    unsigned blockId = blockIdx.x;
    dctTmpChild[ blockId ] = spatInChild[ blockId ] * __cosf( childSpecCoeffNo * d_LUTspat[ blockId ]   );
//     printf( "dctTmpChild [ %d ]: %.2f\n", blockId, dctTmpChild[ blockId ] );
}

__global__ void parentKernel( float* specCoeffNo, float* spatIn, float* dctTmp, double* coeffReduce, float* specOut )       
{
    *specCoeffNo = 1;
    unsigned tmpSpecCoeff = 0;
//     for ( ; *specCoeffNo < 1920; *specCoeffNo += 1 )
//     {
    childKernel<<< 1920, 1 >>>( spatIn, *specCoeffNo, dctTmp );
        //mul const by ( reduce dctTmp )
        for ( unsigned i = 0; i < 1920; i++ )
            *coeffReduce += dctTmp[ i ];        
    tmpSpecCoeff = *specCoeffNo;
    specOut[ tmpSpecCoeff ] = float( *coeffReduce ) * d_sqrConst[ 0 ];
    cudaDeviceSynchronize();
//     }
    printf( "*dctTmp[ 1 ]: %f, \n", dctTmp[ 1 ]);
    printf( "DCT[%i]: %f!\n", tmpSpecCoeff, specOut[ tmpSpecCoeff ] );
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
    float h_sqrtConst = sqrt( 2 / float( N ) ); //constant device variable sqrt(2/N)
    cudaMemcpyToSymbol( d_sqrConst, &h_sqrtConst, sizeof( float ) );
    
    float h_LUTspat[ N ];   //constant device variable LUT spatial argument of freq. cosf()
    for ( unsigned i = 0; i < N; i++ )
        h_LUTspat[ i ] = ( h_PI / float( N ) ) * ( i + 0.5f );
    cudaMemcpyToSymbol( d_LUTspat, &h_LUTspat, N * sizeof( float ) );
    
////note DYNAMIC PARALLELISM FOR EACH - writing kernel:
//  //DCT1D( d_vecD[ 0 ] ) DC freq.
    d_vecD_DCT[ 0 ][ 0 ] = reduce( d_vecD[0].begin(), d_vecD[0].end() ) / float( N );
    cout << "d_vecD_DCT[ 0 ][ 0 ]: " << d_vecD_DCT[ 0 ][ 0 ] << endl;
    cout << "d_vecD_DCT[ 0 ][ 1 ]: " << d_vecD_DCT[ 0 ][ 1 ] << endl;
    
//  //DCT1D( d_vecD[ 0 ] ) AC freq.    
//I) g(m)*__cosf(LUTspat*k)
    float *d_vecD_tmpTransfDCTarray;     //shared memory variable candidate -        note FOR_EACH CANDIDATE also P.S. one Malloc for any
    cudaMalloc( ( void** ) &d_vecD_tmpTransfDCTarray, sizeof( float ) * N );
    thrust::copy( d_vecD_DCT[ 0 ].begin(), d_vecD_DCT[ 0 ].end(), thrust::device_ptr< float >(d_vecD_tmpTransfDCTarray)); //d_vecD_DCT[ 0 ] vector
    float *d_vecD_DCTpriv;     
    cudaMalloc( ( void** ) &d_vecD_DCTpriv, sizeof( float ) * N );
    float *d_tmpSpecCoeffNo;     
    cudaMalloc( ( void** ) &d_tmpSpecCoeffNo, sizeof( double ) * N );
    double *d_tmpCoeffReduce;     
    cudaMalloc( ( void** ) &d_tmpCoeffReduce, sizeof( double ) * N );
    
    float *d_vecD_DCTarray;     
    cudaMalloc( ( void** ) &d_vecD_DCTarray, sizeof( float ) * N );

    parentKernel<<< 1, 1 >>>( d_tmpSpecCoeffNo, d_vecD_tmpTransfDCTarray, d_vecD_DCTpriv, d_tmpCoeffReduce, d_vecD_DCTarray );    //up to 1024 dynPar threads!
    
    
    cudaFree( d_tmpSpecCoeffNo );           //tmp spectrum coefficient number
    cudaFree( d_tmpCoeffReduce );           //sum reduce tmp
    cudaFree( d_vecD_DCTpriv );             //DCT tmp var
    cudaFree( d_vecD_DCTarray );            //DCT 1D result
    cudaFree( d_vecD_tmpTransfDCTarray );   //dynPar deallocation
    cudaFree( d_sqrConst );                 //sqrt 2/N free
    cudaFree( d_LUTspat );                  //spatial LUT
    cudaDeviceSynchronize();
    return 0;
}





