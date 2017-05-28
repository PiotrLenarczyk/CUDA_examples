//STL
#include <iostream>                     //printf, cout, endl
//THRUST
#include <thrust/device_vector.h>       //omit host vectors and costly data transfers
#include <thrust/for_each.h>            //normalize with size inversed FFT
#include <thrust/reduce.h>              //sum array reduce
#include <thrust/sequence.h>            //used in reduce
#include <thrust/execution_policy.h>    //library dependency
#include "device_ptr.h"                 //thrust device_vector to array

using namespace thrust;             //note that default is THRUST!!!

//host globals
unsigned h_BlThKernel[ 4 ];
const float h_PI = 3.1415926535897932384626;
const int ROWY = 1080;       
const int COLX = 1920;
//device variables ( const thread cache; global )
__constant__ float d_LUTrowsY[ ROWY ];//note float lossy quantizer noise as trade-off: speed/accuracy
__constant__ float d_LUTcolsX[ COLX ]; 
__constant__ unsigned d_ROWsY[ 1 ]; //1080
__constant__ unsigned d_COLsX[ 1 ]; //1920
__constant__ unsigned d_BlThKernel[ 4 ]; //ROWY 1080 =8*135; COLX 1920 = 8*240; blocks-thread indexing map
__device__ float d_vecDArray[ ROWY ][ COLX ];       //YX order
__device__ float d_vecDCTArray[ ROWY ][ COLX ];     //YX order
__device__ float d_vecDCT2DArray[ ROWY ][ COLX ];   //YX order
__device__ float d_vecInvDCTArray[ ROWY ][ COLX ];       //YX order
__device__ float d_vecInvDArray[ ROWY ][ COLX ];       //YX order

__global__ void rowsIDCT1D()
{
//                                  ROWS threads no        
    unsigned rowDCT = blockIdx.x * d_BlThKernel[ 1 ] + threadIdx.x; 
    double singleInvArray[ COLX ];
    float singleRow[ COLX ]; //load single row to local variable - omits costly reads from global memory in nested loop
    for ( unsigned xx = 0; xx < COLX; xx++ )
        singleRow[ xx ] = d_vecInvDCTArray[ rowDCT ][ xx ];
    float const_PI = 3.1415926535897932384626 / float( d_COLsX[ 0 ] );  
    float const_DC1D = singleRow[ 0 ] / sqrtf( float( d_COLsX[ 0 ] ) );
    for ( unsigned m = 0; m < COLX; m++ )                                   //nested loops should be parallelised if possible
    {
        singleInvArray[ m ] = 0.0f;
        for ( unsigned xx = 1; xx < COLX; xx++ )
            singleInvArray[ m ] += double( singleRow[ xx ] * __cosf( const_PI * float( xx ) * ( float( m ) + 0.5 ) ) );
        d_vecInvDArray[ rowDCT ][ m ] = const_DC1D + float( singleInvArray[ m ] ) * sqrtf( 2.0f / float( d_COLsX[ 0 ] ) );     
    }
}

__global__ void colsIDCT2D()            //DCT III type - inverse to DCT II type computed without LUT's - used local variables
{
//                                  COLS threads no    
    unsigned colDCT = blockIdx.x * d_BlThKernel[ 3 ] + threadIdx.x;
    double singleAC_InvDCT2D[ ROWY ];
    float singleCol[ ROWY ];
    for ( unsigned yy = 0; yy < ROWY; yy++ )
        singleCol[ yy ] = d_vecDCT2DArray[ yy ][ colDCT ];
    float const_PI = 3.1415926535897932384626 / float( d_ROWsY[ 0 ] );
    float const_DC2D = singleCol[ 0 ] / sqrtf( float( d_ROWsY[ 0 ] ) );
    for ( unsigned m = 0; m < ROWY; m++ )
    {
        singleAC_InvDCT2D[ m ] = 0.0f;
        for ( unsigned yy = 1; yy < ROWY; yy++ )
            singleAC_InvDCT2D[ m ] += double( singleCol[ yy ] * __cosf( const_PI * float( yy ) * ( float( m ) + 0.5 ) ) );
        d_vecInvDCTArray[ m ][ colDCT ] = const_DC2D + float( singleAC_InvDCT2D[ m ] ) * sqrtf( 2.0f / float( d_ROWsY[ 0 ] ) );     
    }
}

//DCT2D GPU kernel
__global__ void colsDCT2D() //DCT2D( colsDCT1D( rowsSignal2DSpatial ) )
{
//                                  COLS threads no        
    unsigned colDCT = blockIdx.x * d_BlThKernel[ 3 ] + threadIdx.x;
    double singleAC_DCT2D[ ROWY ];
    __shared__ double singleDCCol[ COLX ]; //perThread shared memory
    float singleCol[ ROWY ];
    for ( unsigned yy = 0; yy < ROWY; yy++ )
        singleCol[ yy ] = double( d_vecDCTArray[ yy ][ colDCT ] );
    singleDCCol[ colDCT ] = double( singleCol[ 0 ] ); 
    for ( unsigned k = 1; k < ROWY; k++ )
    {
        singleAC_DCT2D[ k ] = 0.0f;
        for ( unsigned yy = 0; yy < ROWY; yy++ )
            singleAC_DCT2D[ k ] += singleCol[ yy ] * __cosf( float( k ) * d_LUTrowsY[ yy ] );
        d_vecDCT2DArray[ k ][ colDCT ] = float( singleAC_DCT2D[ k ] ) * sqrtf( 2.0f / float( d_ROWsY[ 0 ] ) );
        singleDCCol[ colDCT ] += double( singleCol[ k ] );
    }
    d_vecDCT2DArray[ 0 ][ colDCT ] = float( singleDCCol[ colDCT ] ) / sqrtf( float( d_ROWsY[ 0 ] ) );
}

//DCT1D GPU kernel
__global__ void rowsDCT( )                            //rowY steered
{
//                                  ROWS threads no        
    unsigned rowDCT = blockIdx.x * d_BlThKernel[ 1 ] + threadIdx.x;    
    double singleACDCT[ COLX ];
    float singleRow[ COLX ]; //load single row to local variable - omits costly reads from global memory in nested loop
    for ( unsigned xx = 0; xx < COLX; xx++ )
        singleRow[ xx ] = double( d_vecDArray[ rowDCT ][ xx ] );
    for ( unsigned k = 1; k < COLX; k++ )
    {
        singleACDCT[ k ] = 0.0f;
        for ( unsigned xx = 0; xx < COLX; xx++ )
            singleACDCT[ k ] += double( singleRow[ xx ] * __cosf( float( k ) * d_LUTcolsX[ xx ] ) );
        d_vecDCTArray[ rowDCT ][ k ] = float( singleACDCT[ k ] ) * sqrtf( 2.0f / float( d_COLsX[ 0 ] ) );        
    }
    d_vecDCTArray[ rowDCT ][ 0 ] = reduce( seq, d_vecDArray[ rowDCT ], d_vecDArray[ rowDCT ] + d_COLsX[ 0 ] ) / sqrtf( float( d_COLsX[ 0 ] ) );     //DC DCT1D computed with thrust reduce for rows
}

//populate 1D array to 2D via Dynamic Parallelism ( cols on rows )
__global__ void populateColsKernel( unsigned childRow, float* d_vecD_tmpTransfarray )
{
//                                  COLX threads no    
    unsigned colInd = blockIdx.x * d_BlThKernel[ 3 ] + threadIdx.x;
    d_vecDArray[ childRow ][ colInd ] = d_vecD_tmpTransfarray[ childRow * d_COLsX[ 0 ] + colInd ];
}
__global__ void populateRowsKernel( float* d_vecD_tmpTransfarray )
{
//                                  ROWS threads no    
    unsigned rowInd = blockIdx.x * d_BlThKernel[ 1 ] + threadIdx.x;
    populateColsKernel<<< d_BlThKernel[ 2 ], d_BlThKernel[ 3 ] >>>( rowInd, d_vecD_tmpTransfarray );
}
//print inversed array elements kernel
__global__ void printInvDArray()
{
    unsigned y = blockIdx.x; unsigned x = threadIdx.x;
    printf( "d_vecInvDArray[y=%i][x=%i]: %f \n", y, x, d_vecInvDArray[ y ][ x ] );
}

//print inversed array elements kernel
__global__ void printInvDCTArray()
{
    unsigned y = blockIdx.x; unsigned x = threadIdx.x;
    printf( "d_vecInvDCTArray[y=%i][x=%i]: %f \n", y, x, d_vecInvDCTArray[ y ][ x ] );
}

//print DCT1D array elements kernel
__global__ void printDCT1D()
{
    unsigned y = blockIdx.x; unsigned x = threadIdx.x;
    printf( "d_vecDCTArray[y=%i][x=%i]: %f \n", y, x, d_vecDCTArray[ y ][ x ] );
}

//print DCT2D array elements kernel
__global__ void printDCT2D()
{
    unsigned y = blockIdx.x; unsigned x = threadIdx.x;
    printf( "d_vecDCT2DArray[y=%i][x=%i]: %f \n", y, x, d_vecDCT2DArray[ y ][ x ] );
}

int main()
{
////    THRUST example data ( not ideal - should be used 1D array, but it is less intuitive, especially multidimensional indexing )
    device_vector< float > dim1( COLX ); sequence( device, dim1.begin(), dim1.end(), 0.1f );   
//  default data container thrust::device_vector< float >[ 2D ]; 
    device_vector< float > d_vecD[ ROWY ]; 
    for ( unsigned i = 0; i < ROWY; i++ )       //vecD must be init! - note that vecD is not neccessary to be a constant size via 1D ( sometimes useful for sparse data! )
        d_vecD[ i ] = dim1;
    
//  populate each GPUThread constant cache variables
    float h_LUTcolsX[ COLX ];   //constant device variable LUT spatial argument of rows freq. __cosf() - const part of cosine transform base vectors
    float h_LUTrowsY[ ROWY ];   //constant device variable LUT spatial argument of cols freq. __cosf() - const part of cosine transform base vectors
    for ( unsigned i = 0; i < COLX; i++ )
        h_LUTcolsX[ i ] = ( h_PI / float( COLX ) ) * ( i + 0.5f );
    for ( unsigned i = 0; i < ROWY; i++ )
        h_LUTrowsY[ i ] = ( h_PI / float( ROWY ) ) * ( i + 0.5f );
    cudaMemcpyToSymbol( d_LUTcolsX, &h_LUTcolsX, sizeof( float ) * COLX );
    cudaMemcpyToSymbol( d_LUTrowsY, &h_LUTrowsY, sizeof( float ) * ROWY );
    
//  device size variables
//   ROWY 1080 = [0]8     *             [1]135;     COLX 1920 =   [2]8        *        [3]240
    h_BlThKernel[ 0 ] = 8; h_BlThKernel[ 1 ] = 135; h_BlThKernel[ 2 ] = 8; h_BlThKernel[ 3 ] = 240;    
    cudaMemcpyToSymbol( d_BlThKernel, &h_BlThKernel, sizeof( unsigned ) * 4 );
    cudaMemcpyToSymbol( d_ROWsY, &ROWY, sizeof( unsigned ) * 1 ); cudaMemcpyToSymbol( d_COLsX, &COLX, sizeof( unsigned ) * 1 );

//  //DCT1D( d_vecD in GPUmemory copy via rows )
    float *d_vecD_tmpTransfarray;
    cudaMalloc( ( void** ) &d_vecD_tmpTransfarray, sizeof( float ) * ROWY * COLX );
    for ( unsigned i = 0; i < ROWY; i++ )
        copy( device, d_vecD[ i ].begin(), d_vecD[ i ].end(), device_ptr< float >( &d_vecD_tmpTransfarray[ i * COLX ] ) );
    
//             ROWS = 1080: ROWSBl = 8          ROWSTh = 135    
    populateRowsKernel<<< h_BlThKernel[ 0 ], h_BlThKernel[ 1 ] >>>( d_vecD_tmpTransfarray );    //populate 2D array from temporary 1D
    cudaFree( d_vecD_tmpTransfarray );                                                          //free temporary 1D
//             ROWS = 1080: ROWSBl[0] = 8      ROWSTh[1] = 135        
    rowsDCT<<< h_BlThKernel[ 0 ], h_BlThKernel[ 1 ] >>>();                                //DCT 1D via rows computed from definition - no methods enhancement
//                          COLX 1920 =   [2]8   *        [3]240    
    colsDCT2D<<< h_BlThKernel[ 2 ], h_BlThKernel[ 3 ] >>>();                              //DCT 2D via DCT1D cols
//                          COLX 1920 =   [2]8   *        [3]240    
    colsIDCT2D<<< h_BlThKernel[ 2 ], h_BlThKernel[ 3 ] >>>(); 
    rowsIDCT1D<<< h_BlThKernel[ 0 ], h_BlThKernel[ 1 ] >>>();
//               nY  nX
    printDCT1D<<< 3, 3 >>>();                                                                  //print DCT1D results
//               nY  nX
    printDCT2D<<< 3, 3 >>>();                                                                   //print DCT2D results
//               nY  nX
    printInvDCTArray<<< 3, 3 >>>();
//               nY  nX
    printInvDArray<<< 3, 3 >>>();
    
//  free gpu memory     
    cudaFree( d_ROWsY );                    //rowsY device constant
    cudaFree( d_COLsX );                    //colsX device constant
    cudaFree( d_BlThKernel );               //Blocks & Threads organise for row-col and col-row access
    cudaFree( d_LUTcolsX );                 //spatial LUT
    cudaFree( d_LUTrowsY );                 //spatial LUT
    cudaDeviceSynchronize();
    return 0;
}
//P.S. rows 1080 blocks could be called directly - few times slower in comparision to <<<8_Blocks,135_Threads>>> cause hardware organisation,
//P.P.S. note Dynamic Parallelism could efficiently compute up to 5D nested access to data in parallel ( unless N(1:5)^5 < 2^31-1 blocks ),
//P.P.P.S. note more computations in single thread is less costly than additional stage of Dynamic Parallelism - there is a trade-off ( #sudo nvprof ./a.out ),
//P.P.P.P.S DCT2D cols( DCT1D( dataRows ) ) is equivalent to DCT2D rows( DCT1D( dataCols ) ). Results are not too accurate ( adding small to large float errors accumulation ) for DCT2D transform - better approach would be to extract DCT from cuFFT spectrum part ( with constant multiplication ).
