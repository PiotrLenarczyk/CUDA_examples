//STL
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <vector>
#include <iterator>
#include <time.h>
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
__constant__    unsigned    d_ROWY[ 1 ];    //const ROWY
__constant__    unsigned    d_COLX[ 1 ];    //const COLX
__constant__    unsigned    d_matsNo[ 1 ];
__constant__    unsigned    d_matsSizes[ MATNo ];//image dependend
__constant__    unsigned    d_ROWsY[ MATNo ];//image dependend
__constant__    unsigned    d_COLsX[ MATNo ];//image dependend
__constant__    unsigned    d_BlThKernel[ 2 ];                 // 16 Blocks; 256 Threads = 4096 Threads per dimension 
__device__      float       d_vecDArray[ MATNo ][ ROWY ][ COLX ];       //images: matZYX order
__device__      float       d_vecD_tmpTransfarray[ MATNo * ROWY * COLX ]; //images: mat1D[0]mat1D[1]mat1D[2]...
__global__      void        syncKernel(){}

__global__ void populateMatsRowsColsKernel() 
{//convert global1D indexing to local YX indexing of image matrix[ Z ] in 3D space of d_vecDArray[Z][Y][X]
    unsigned i = 0, matZ = 0, matY = 0, matX = 0;
    unsigned globalInd = blockIdx.x * d_BlThKernel[ 1 ] + threadIdx.x;
    unsigned localInd = globalInd;
    
    unsigned acc = d_matsSizes[ 0 ];
    for ( i = 0; i < d_matsNo[ 0 ]; i++ )
        if ( globalInd >= acc )
            {
                matZ++;
                acc += d_matsSizes[ i + 1 ];
                localInd -= d_matsSizes[ i ];
            }
            
    matY = localInd / d_COLsX[ matZ ];
    matX = localInd - matY * d_COLsX[ matZ ];
    d_vecDArray[ matZ ][ matY ][ matX ] = d_vecD_tmpTransfarray[ globalInd ];
}

__global__ void populateColsKernel( unsigned childMatInd, unsigned childRow )
{//                                  COLX threads no    
    unsigned colInd = blockIdx.x * d_BlThKernel[ 0 ] + threadIdx.x;
    unsigned rowColTmp;
    if ( childMatInd != 0 ) //scaling index to current matrix via prior matrices YX's
        rowColTmp = childMatInd - 1;
    unsigned tmpInd = childRow * d_COLsX[ childMatInd ] + colInd +
                                       childMatInd * d_ROWsY[ rowColTmp ] * d_COLsX[ rowColTmp ];
        d_vecDArray[ childMatInd ][ childRow ][ colInd ] = d_vecD_tmpTransfarray[ tmpInd ];
}
__global__ void populateRowsKernel( unsigned MatInd )
{//                                  ROWS threads no    
    unsigned rowInd = blockIdx.x * d_BlThKernel[ 0 ] + threadIdx.x;
    populateColsKernel<<< d_BlThKernel[ 0 ], d_BlThKernel[ 1 ] >>>( MatInd, rowInd );
}

__global__ void populateMatsKernel()
{
    unsigned matInd = blockIdx.x;
    populateRowsKernel<<< d_BlThKernel[ 0 ], d_BlThKernel[ 1 ] >>>( matInd );
}

__global__ void testd_vecD_tmpTransfarray()
{//test values:
    printf( "d_vecD_tmpTransfarray[0][0][0]: %f\n", d_vecD_tmpTransfarray[ 0 ] );
    printf( "d_vecD_tmpTransfarray[0][0][1]: %f\n", d_vecD_tmpTransfarray[ 1 ] );
    printf( "d_vecD_tmpTransfarray[0][0][2]: %f\n", d_vecD_tmpTransfarray[ 2 ] );
    printf( "d_vecD_tmpTransfarray[1][0][0]: %f\n", d_vecD_tmpTransfarray[ 262144 ] );
    printf( "d_vecD_tmpTransfarray[1][0][1]: %f\n", d_vecD_tmpTransfarray[ 262144 + 1 ] );
    printf( "d_vecD_tmpTransfarray[1][0][2]: %f\n", d_vecD_tmpTransfarray[ 262144 + 2 ] );
    printf( "d_vecD_tmpTransfarray[2][0][0]: %f\n", d_vecD_tmpTransfarray[ 262144 + 262144 ] );
    printf( "d_vecD_tmpTransfarray[2][0][1]: %f\n", d_vecD_tmpTransfarray[ 262144 + 262144 + 1 ] );
    printf( "d_vecD_tmpTransfarray[2][0][2]: %f\n", d_vecD_tmpTransfarray[ 262144 + 262144 + 2 ] );
}

__global__ void print()
{
    printf( "d_matsNo: %i\n", d_matsNo[ 0 ] );
    printf( "blocks: %i\n", d_BlThKernel[ 0 ] );
    printf( "threads: %i\n", d_BlThKernel[ 1 ] );
    for ( int i = 0; i < d_matsNo[ 0 ]; i++ )
    {
        printf( "mat[%i]: \n", i );
        printf( "d_matsSizes[%i]: %i [", i, d_matsSizes[ i ] );
        printf( " d_ROWsY=%i x", d_ROWsY[ i ] );
        printf( " d_COLsX=%i ]\n", d_COLsX[ i ] );
        printf( "d_vecDArray[ %i ][ 0 ][ 0 ]: %f\n", i, d_vecDArray[ i ][ 0 ][ 0 ] );
        printf( "d_vecDArray[ %i ][ 0 ][ 1 ]: %f\n", i, d_vecDArray[ i ][ 0 ][ 1 ] );
        printf( "d_vecDArray[ %i ][ 0 ][ 2 ]: %f\n", i, d_vecDArray[ i ][ 0 ][ 2 ] );
    }
    printf( "\n\n\n" );
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
    
//  test values:
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
    
    unsigned h_matsSizes[ h_matNo ];
    for ( i = 0; i < h_matNo; i++ ) h_matsSizes[ i ] = h_ROWsY[ i ] * h_COLsX[ i ];
    cudaMemcpyToSymbol( d_COLX, &COLX, sizeof( unsigned ) );
    cudaMemcpyToSymbol( d_ROWY, &ROWY, sizeof( unsigned ) );
    cudaMemcpyToSymbol( d_matsNo, &h_matNo, sizeof( unsigned ) );
    cudaMemcpyToSymbol( d_matsSizes, &h_matsSizes, sizeof( unsigned ) * h_matNo );
    cudaMemcpyToSymbol( d_ROWsY, &h_ROWsY, sizeof( unsigned ) * h_matNo );
    cudaMemcpyToSymbol( d_COLsX, &h_COLsX, sizeof( unsigned ) * h_matNo );
    cudaMemcpyToSymbol( d_BlThKernel, &h_BlThKernel, sizeof( unsigned ) * 2 );
    cudaMemcpyToSymbol( d_vecD_tmpTransfarray, &vecDIn[ 0 ], sizeof( float ) * vecDIn.size() );
    testd_vecD_tmpTransfarray<<< 1, 1 >>>();
    cudaDeviceSynchronize();
    
    cout << "================== GPU ======================" << endl;
    clock_t t = clock();
    unsigned blocksMat = 0;
    for ( i = 0; i < h_matNo; i++ )
    {
        blocksMat += h_ROWsY[ i ] * h_COLsX[ i ];
    }
//                                  podzielnosc przez h_BlThKernel
    cout << "1D blocks " <<  blocksMat / h_BlThKernel[ 1 ] << endl;
    populateMatsRowsColsKernel<<< blocksMat / h_BlThKernel[ 1 ],  h_BlThKernel[ 1 ] >>>();
    cudaDeviceSynchronize();
    cout << "1D kernel CPU clocks: " << clock() - t << endl;
    print<<<1,1>>>();

    t = clock();
    for ( i = 0; i < h_matNo; i++ )
        populateRowsKernel<<< h_BlThKernel[ 0 ], h_BlThKernel[ 1 ] >>>( i ); //dynamic parallelism 2 stages ( via 2D )
    cudaDeviceSynchronize();
    cout << "sequential DynPar 2D kernels CPU clocks: " << clock() - t << endl;
    print<<<1,1>>>();

    t = clock();
    populateMatsKernel<<< 3, 1 >>>();    //dynamic parallelism 3 stages ( via 3D )
    cudaDeviceSynchronize();
    cout << "DynPar 3D kernels CPU clocks: " << clock() - t << endl;
    print<<<1,1>>>();

    std::system( "nvidia-smi" );
    
//  free gpu memory     
    cudaFree( d_ROWY );
    cudaFree( d_COLX );
    cudaFree( d_matsNo );
    cudaFree( d_matsSizes );
    cudaFree( d_ROWsY );                    //rowsY device constant
    cudaFree( d_COLsX );                    //colsX device constant
    cudaFree( d_BlThKernel );               //Blocks & Threads organise for row-col and col-row access
    cudaFree( d_vecDArray );
    cudaFree( d_vecD_tmpTransfarray );
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}

//P.S. and again dynamic parallelism not so fast as expected for 3D data structures in type[][][] manner ( pointers to pointers? ).
