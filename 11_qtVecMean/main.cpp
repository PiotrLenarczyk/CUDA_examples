//header
#include "HOST_GPU.h"

//tmp HOST + GPU GLOBALS
vector < int > acc( 12, 4 );
vector < vector < int > > acc2( 200 );
vector < int > accRow( 100, 19 );
//HOST + GPU GLOBALS
vector < float > firstVec( N, 3.14f );
vector < float > secondVec( N, 2.72f );
vector < float > resultsHost( N );
vector < float > resGPU( N, 0.0f );
vector < vector < float > > firstMat( N );
vector < vector < float > > secondVecMat( N );
vector < vector < float > > resultsHostMat( N );
vector < vector< float > > resultsGPU( N );
float tmp = 0.0f;

/////////////////////////////// MAIN //////////////////////////////////////////
int main(int argc, char *argv[])
{
    for ( int  i = 0; i < 200; i++ )
        acc2[ i ] = accRow;
   vector< cudaError_t > cuerr;
   int it = 10;
   cuerr.push_back( cuda_main( it ) );
   if ( cuerr.at( cuerr.size() - 1 ) != cudaSuccess)
       cout << "CUDA Error: " << cudaGetErrorString( cuerr.at( cuerr.size() - 1 ) ) << endl;


    for ( int i = 0; i < N; i++ )				//basic CPU processing
    {
        firstMat[ i ] = firstVec;   //vector < vector < float > > firstMat( N, firstVec );
        secondVecMat[ i ] = secondVec;
        resultsGPU[ i ] = resGPU;
        resultsHostMat[ i ] = resultsHost;
    }
    clock_t t;
    if ( N < 9000 )
    {
        t = clock();
        float sumVecA = 0;
        float sumVecB = 0;
        for ( int j = 0; j < N; j++ )
        {
            for ( int i = 0; i < N; i++ )
            {
                sumVecA += firstMat[ j ][ i ];
                sumVecB += secondVecMat[ j ][ i ];
            }
            resultsHostMat[ j ][ 0 ] = sumVecA / N;
            resultsHostMat[ j ][ 1 ] = sumVecB / N;
            sumVecA = 0; sumVecB = 0;
        }
        cout << "Host sequential computations time: " << ((float)(clock() - t))/CLOCKS_PER_SEC << "[s] ( g++ )" << endl;
    }
    
    t = clock();
    memInit();
    vector < thread > gpuAsync3( N );
    for ( int i = 0; i < N; i++ )
    {
        int *iPtr = &( i );
        gpuAsync3[ i ] = thread( dodajMatJadro, iPtr );
        gpuAsync3[ i ].join();
    }
    memFree();
    cout << "time of Async (single join() + basic OPTO) vec<vec<>> GPU time: " << ((float)(clock() - t))/CLOCKS_PER_SEC << "[s]" << endl;

    for ( int i = 7; i < 8; i += 3 )
    {
        cout << "resultsHostMat[ " << i << " ][ 0 ]: " << resultsHostMat[ i ][ 0 ] << endl;
        cout << "resultsGPU[ " << i << " ][ 0 ]: " << resultsGPU[ i ][ 0 ] << endl;
        cout << "resultsGPU[ " << i << " ][ 1 ]: " << resultsGPU[ i ][ 1 ] << endl;
    }

    return 0;
}
