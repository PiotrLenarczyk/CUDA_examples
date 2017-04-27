//STL
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
//THRUST
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>       
#include <thrust/for_each.h>            
#include <thrust/reduce.h>              
#include <thrust/sequence.h>            
#include <thrust/execution_policy.h>    

using namespace thrust;
using std::cout; using std::endl; using std::cerr;

//host globals
const unsigned ROWY = 4096;       //4K oversizing; 1.2GB ~ 20matrices
const unsigned COLX = 4096; 
unsigned h_ROWsY, h_COLsX, h_matNo;
const unsigned h_BlThKernel[ 2 ] = { 16, 256 };
//device variables
__constant__    unsigned    d_ROWsY[ 1 ];
__constant__    unsigned    d_COLsX[ 1 ];
__constant__    unsigned    d_BlThKernel[ 1 ];                 // 16 Blocks; 256 Threads = 4096 Threads per dimension 
__device__      float       d_vecDArray[ ROWY ][ COLX ];       //YX order
__device__      float       d_vecD_tmpTransfarray[ ROWY * COLX ];

__global__ void populateColsKernel( unsigned childRow )
{
//                                  COLX threads no    
    unsigned colInd = blockIdx.x * d_BlThKernel[ 0 ] + threadIdx.x;
    d_vecDArray[ childRow ][ colInd ] = d_vecD_tmpTransfarray[ childRow * d_COLsX[ 0 ] + colInd ];
}
__global__ void populateRowsKernel( )
{
//                                  ROWS threads no    
    unsigned rowInd = blockIdx.x * d_BlThKernel[ 0 ] + threadIdx.x;
    populateColsKernel<<< d_BlThKernel[ 0 ], d_BlThKernel[ 1 ] >>>( rowInd );
}

__global__ void printConst()
{
    printf( "d_ROWsY: %i\n", d_ROWsY[ 0 ] );
    printf( "d_COLsX: %i\n", d_COLsX[ 0 ] );
    printf( "blocks: %i\n", d_BlThKernel[ 0 ] );
    printf( "threads: %i\n", d_BlThKernel[ 1 ] );
    for ( int i = 0; i < 3; i++ )
        printf( "d_vecDArray[ 0 ][ %i ]: %f\n", i, d_vecDArray[ 0 ][ i ] );
}

int main( int argc, char* argv[] )
{
    cout << std::setprecision( 9 );
    //load matrices to GPU:
    std::ifstream data_fileIn;      // NOW it's ifstream
    data_fileIn.open( "imFile.txt", std::ios::in | std::ios::binary );
    data_fileIn.read( reinterpret_cast< char* >( &h_matNo ), 1 * sizeof( unsigned ) );
    data_fileIn.read( reinterpret_cast< char* >( &h_ROWsY ), 1 * sizeof( unsigned ) );
    data_fileIn.read( reinterpret_cast< char* >( &h_COLsX ), 1 * sizeof( unsigned ) );
    cout << h_matNo << " images: [ " << h_ROWsY << " x " << h_COLsX << " ]:" << endl;
    host_vector < float > vecIn( h_ROWsY * h_COLsX );
    data_fileIn.read( reinterpret_cast< char* >( &vecIn[ 0 ] ), h_ROWsY * h_COLsX * sizeof( float ) );
    data_fileIn.close();
    
    for ( int i = 0; i < 3; i++ )
        cout << "vecIn[ " << i << " ]: " << vecIn[ i ] << endl;
    cout << "================== GPU ======================" << endl;
    cudaMemcpyToSymbol( d_ROWsY, &h_ROWsY, sizeof( unsigned ) );
    cudaMemcpyToSymbol( d_COLsX, &h_COLsX, sizeof( unsigned ) );
    cudaMemcpyToSymbol( d_BlThKernel, &h_BlThKernel, sizeof( unsigned ) * 2 );
    cudaMemcpyToSymbol( d_vecD_tmpTransfarray, &vecIn[0], sizeof( float ) * h_ROWsY * h_COLsX );
    populateRowsKernel<<< h_BlThKernel[ 0 ], h_BlThKernel[ 1 ] >>>( );
    printConst<<<1,1>>>();
    std::system( "nvidia-smi" );
    
//  free gpu memory     
    cudaFree( d_ROWsY );                    //rowsY device constant
    cudaFree( d_COLsX );                    //colsX device constant
    cudaFree( d_BlThKernel );               //Blocks & Threads organise for row-col and col-row access
    cudaFree( d_vecDArray );
    cudaFree( d_vecD_tmpTransfarray );
    
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}
