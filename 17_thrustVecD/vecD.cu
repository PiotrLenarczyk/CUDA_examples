//STL
#include <cstdlib>                  //atol
#include <iostream>                 //printf, cout, endl
//CUDA
#include <thrust/device_vector.h>   //omit host vectors and costly data transfers

using namespace thrust;             //note that default is THRUST!!!
using std::cout; using std::endl;   //STL is for data viewing

long int N( 0 );

//only applicable for profits from big enough data ( ~N=8000 ) parallel processing
int main( int argc, char *argv[] )  
{
    clock_t t( clock() );
    N = atol( argv[ argc - 1 ] ); 
    if ( N == 0 ) 
    {
        cout << "There are no input arguments!" << endl;
        cudaDeviceSynchronize();
        return 0;
    }
    else
        printf( "N = %ld\n", N );
    
////    C example:
	int ROW = 2;
	int COL = N;
	float **hostPtrVal = new float* [ ROW ]; //matrix rowsY
	for ( int i = 0; i < ROW; i++ )
		hostPtrVal[ i ] = new float[ COL ];
	hostPtrVal[ 0 ][ 1 ] = 11.1f;
	cout << "hostPtrVal[0][1]: " << hostPtrVal[ 0 ][ 1 ] << endl;
	for ( int i = 0; i < ROW; i++ )
		delete[] hostPtrVal[ i ];
	delete[] hostPtrVal;
    
////    THRUST example        
    int ROWY = 2;
    int COLX = N;
    device_vector< float > dim1( COLX, 0.1f );
    device_vector< float > d_vecD[ ROWY ];
    for ( unsigned i = 0; i < ROWY; i++ )
        d_vecD[ i ] = dim1;
    d_vecD[ 0 ][ 1 ] = 11.1f;
    cout << "d_vecD[ 0 ][ 1 ]: " << d_vecD[ 0 ][ 1 ] << endl;
    cout << "d_vecD_Row[0].size(): " << int( device_vector< float >( d_vecD[ 0 ] ).size() ) << endl;
//  P.S. note rare sparse matrices processing - or buy ArrayFire
    
////    THRUST vecD example
    int timeROWY = 2;
    int timeCOLX = N;
    unsigned time = 10000;
    device_vector< float > dim1D( timeCOLX, 0.1f );
    device_vector< float > d_vecTimeD[ time ][ timeROWY ];
    for ( unsigned i = 0; i < timeROWY; i++ )
        d_vecTimeD[ 0 ][ i ] = dim1;
    d_vecTimeD[ 0 ][ 0 ][ 1 ] = 11.1f;
    cout << "d_vecTimeD[ zeroTime ][ 0 ][ 1 ]: " << d_vecTimeD[ 0 ][ 0 ][ 1 ] << endl;
    cout << "d_vecTimeD_time[0]_Row[0].size(): " << int( device_vector< float >( d_vecTimeD[ 0 ][ 0 ] ).size() ) << endl;
//  P.S. note quite similar dimensional dependencies to ordinary digital movie ( as 3D signal )
    
	return 0;
}
