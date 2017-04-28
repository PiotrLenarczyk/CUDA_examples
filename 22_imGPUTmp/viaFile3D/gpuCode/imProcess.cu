//STL
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <vector>
#include <iterator>
//THRUST
#include <thrust/host_vector.h>         
#include <thrust/device_vector.h>       
#include <thrust/for_each.h>            
#include <thrust/reduce.h>              
#include <thrust/sequence.h>            
#include <thrust/execution_policy.h>    
#include <device_ptr.h>                 

using namespace thrust;
using std::cout; using std::endl; using std::cerr; using std::vector;

//host globals
const unsigned ROWY = 4096;       //4K oversizing; .64GB ~ 10matrices
const unsigned COLX = 4096;
const unsigned MATNo = 10;
unsigned i, h_matNo;
const unsigned h_BlThKernel[ 2 ] = { 16, 256 };
//device variables
__constant__    unsigned    d_matsNo[ MATNo ];
__constant__    unsigned    d_ROWsY[ MATNo ];
__constant__    unsigned    d_COLsX[ MATNo ];
__constant__    unsigned    d_BlThKernel[ 2 ];                 // 16 Blocks; 256 Threads = 4096 Threads per dimension 
__device__      float       d_vecDArray[ MATNo ][ ROWY ][ COLX ];       //matYX order
__device__      float       d_vecD_tmpTransfarray[ MATNo * ROWY * COLX ];

__global__ void populateColsKernel( unsigned childMatInd, unsigned childRow )
{//                                  COLX threads no    
    unsigned colInd = blockIdx.x * d_BlThKernel[ 0 ] + threadIdx.x;
    d_vecDArray[ childMatInd ][ childRow ][ colInd ] = 
                d_vecD_tmpTransfarray[ childRow * d_COLsX[ childMatInd ] + colInd + 
                                       childMatInd * d_ROWsY[ childMatInd ] * d_COLsX[ childMatInd ] ];
}
__global__ void populateRowsKernel( unsigned MatInd )
{//                                  ROWS threads no    
    unsigned rowInd = blockIdx.x * d_BlThKernel[ 0 ] + threadIdx.x;
    populateColsKernel<<< d_BlThKernel[ 0 ], d_BlThKernel[ 1 ] >>>( MatInd, rowInd );
}

__global__ void printConst()
{
    printf( "d_matsNo: %i\n", d_matsNo[ 0 ] );
    printf( "blocks: %i\n", d_BlThKernel[ 0 ] );
    printf( "threads: %i\n", d_BlThKernel[ 1 ] );
    for ( int i = 0; i < 3; i++ )
    {
        printf( "mat[%i]: ", i );
        printf( " d_ROWsY: %i", d_ROWsY[ i ] );
        printf( " d_COLsX: %i\n", d_COLsX[ i ] );
        printf( "d_vecDArray[ %i ][ 0 ][ 0 ]: %f\n", i, d_vecDArray[ i ][ 0 ][ 0 ] );
        printf( "d_vecDArray[ %i ][ 0 ][ 1 ]: %f\n", i, d_vecDArray[ i ][ 0 ][ 1 ] );
        printf( "d_vecDArray[ %i ][ 0 ][ 2 ]: %f\n", i, d_vecDArray[ i ][ 0 ][ 2 ] );
    }
}

__global__ void print1D()
{
    for ( int i = 0; i < 3; i++ )
        printf( "d_vecD_tmpTransfarray[%i]: %f\n", i, d_vecD_tmpTransfarray[ i ] );
    for ( int i = 0; i < 3; i++ )
        printf( "d_vecD_tmpTransfarray[%i]: %f\n", 262144 + i, d_vecD_tmpTransfarray[ 262144 + i ] );
    for ( int i = 0; i < 3; i++ )
        printf( "d_vecD_tmpTransfarray[%i]: %f\n", 262144 + 262144 + i, d_vecD_tmpTransfarray[ 262144 + 262144 + i ] );
}

int main( int argc, char* argv[] )
{
    cout << std::setprecision( 9 );
    //load matrices to GPU:
    std::ifstream data_fileIn;     
    data_fileIn.open( "imFile.txt", std::ios::in | std::ios::binary );
    if ( !data_fileIn.is_open() ) { cerr << "File open error!\n"; return -1; }
    data_fileIn.read( reinterpret_cast< char* >( &h_matNo ), 1 * sizeof( unsigned ) );
    if ( h_matNo > MATNo ) { cerr << "Memory lack!\n"; return -1; }
    cout << "images: " << h_matNo << endl;
    unsigned h_ROWsY[ h_matNo ], h_COLsX[ h_matNo ];
    vector < float > vecDIn;
    for ( i = 0; i < h_matNo; i++ )
    {
        data_fileIn.read( reinterpret_cast< char* >( &h_ROWsY[ i ] ), sizeof( unsigned ) * 1 );
        data_fileIn.read( reinterpret_cast< char* >( &h_COLsX[ i ] ), sizeof( unsigned ) * 1 );
        vector < float > vecIn( h_ROWsY[ i ] * h_COLsX[ i ] );
        data_fileIn.read( reinterpret_cast< char* >( &vecIn[ 0 ] ), sizeof( float ) * h_ROWsY[ i ] * h_COLsX[ i ] );
        vecDIn.insert( std::end( vecDIn ), std::begin( vecIn ), std::end( vecIn ) );          //vector append data
        cout << " image[" << i << "]: [ " << h_ROWsY[ i ] << " x " << h_COLsX[ i ] << " ].size(): " << vecIn.size() << endl;
        cout << "vecIn[ " << i << " ][ " << vecIn[ 0 ] << " ]" << endl;
    }
    data_fileIn.close();
/*
imY[ 0 ][ 0 ]: 154.162
imY[ 0 ][ 1 ]: 155.021
imY[ 0 ][ 2 ]: 156.136
imY[ 1 ][ 0 ]: 140.684
imY[ 1 ][ 1 ]: 63.9482
imY[ 1 ][ 2 ]: 57.9158
imY[ 2 ][ 0 ]: 22.6233
imY[ 2 ][ 1 ]: 22.6233
imY[ 2 ][ 2 ]: 22.9316
 */   
    cout << "vecDIn.size(): " << vecDIn.size() << endl;
    cout << "vecDIn[0][0]: " << vecDIn[ 0 ] << endl;
    cout << "vecDIn[0][1]: " << vecDIn[ 1 ] << endl;
    cout << "vecDIn[0][2]: " << vecDIn[ 2 ] << endl;
    cout << "vecDIn[1][0]: " << vecDIn[ 262144 ] << endl;
    cout << "vecDIn[1][1]: " << vecDIn[ 262144 + 1 ] << endl;
    cout << "vecDIn[1][2]: " << vecDIn[ 262144 + 2 ] << endl;
    cout << "vecDIn[2][0]: " << vecDIn[ 262144 + 262144 ] << endl;
    cout << "vecDIn[2][1]: " << vecDIn[ 262144 + 262144 + 1 ] << endl;
    cout << "vecDIn[2][2]: " << vecDIn[ 262144 + 262144 + 2 ] << endl;

    
    cout << "================== GPU ======================" << endl;
    cudaMemcpyToSymbol( d_matsNo, &h_matNo, sizeof( unsigned ) );
    cudaMemcpyToSymbol( d_ROWsY, &h_ROWsY, sizeof( unsigned ) * h_matNo );
    cudaMemcpyToSymbol( d_COLsX, &h_COLsX, sizeof( unsigned ) * h_matNo );
    cudaMemcpyToSymbol( d_BlThKernel, &h_BlThKernel, sizeof( unsigned ) * 2 );
    cudaMemcpyToSymbol( d_vecD_tmpTransfarray, &vecDIn[ 0 ], sizeof( float ) * vecDIn.size() );
    print1D<<< 1, 1 >>>();

    
    
    blad dla trzeciego zdjecia
    
    
    
//     for ( i = 0; i < h_matNo; i++ )
        populateRowsKernel<<< h_BlThKernel[ 0 ], h_BlThKernel[ 1 ] >>>( int(2) );
    printConst<<<1,1>>>();
//     std::system( "nvidia-smi" );
    
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

//P.S. and again dynamic parallelism not so fast as expected for 3D data.
