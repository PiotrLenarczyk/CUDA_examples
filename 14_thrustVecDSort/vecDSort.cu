//STL
#include <cstdlib>                  //atol
#include <iostream>                 //printf, cout, endl
#include <time.h>                   //timings
#include <vector>					//vector for example host picture
#include <algorithm>                //std::copy
//CUDA
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>   //omit host vectors and costly data transfers
#include <thrust/sort.h>            //sorting
#include <thrust/random.h>          //random example data generator
#include <thrust/tuple.h>           //abstract data vectors

using namespace thrust;             //note that default is THRUST!!!
using std::cout; using std::endl;   //STL is for timings and data viewing
using std::vector;

//http://stackoverflow.com/questions/26676806/efficiency-of-cuda-vector-types-float2-float3-float4

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
    
    default_random_engine rng( 18612 );
    uniform_real_distribution< float > dist_f( 0.0f, 0.996f );
    device_vector< float > vecValRef( N ); float valTmp( 0 );
    typedef pair< float, unsigned int > pairFloatInd; //< float, index > pair
    device_vector< pairFloatInd > vecXValInd( N );
    device_vector< pairFloatInd > vecYValInd( N ); 
    device_vector< pairFloatInd > vecZValInd( N ); 
    device_vector< pairFloatInd > vecValInd( N ); 
    for ( unsigned int i = 0; i < N; i++ ) //such sequential data generation on GPU is slow
    {
        vecXValInd[ i ] = pairFloatInd( dist_f( rng ), i );
        vecYValInd[ i ] = pairFloatInd( dist_f( rng ), i );
        vecZValInd[ i ] = pairFloatInd( dist_f( rng ), i );
        valTmp = 5.1f * dist_f( rng );
        vecValRef[ i ] = valTmp;
        vecValInd[ i ] = pairFloatInd( valTmp, i );
    }
    
    for ( unsigned int i = 0; i < N; i++ )
    {
        printf( "vec[ index ][ X, Y, Z ][ val ]: [ %02d ][ %.3f, %.3f, %.3f ][ %.3f ]\n",
                i,
                float( pairFloatInd( vecXValInd[ i ] ).first ),
                float( pairFloatInd( vecYValInd[ i ] ).first ),
                float( pairFloatInd( vecZValInd[ i ] ).first ),
                float( pairFloatInd( vecValInd[ i ] ).first )
              );
    }
    
    t = clock();
    sort( vecXValInd.begin(), vecXValInd.end() );       //zdies speedup
    sort( vecYValInd.begin(), vecYValInd.end() );
    sort( vecZValInd.begin(), vecZValInd.end() );
    sort( vecValInd.begin(), vecValInd.end() );
    cout << "time: " << 1000 * float( clock() - t ) / CLOCKS_PER_SEC << "[ ms ]" << endl;
    cout << "Dimensionally sorted vecD:" << endl;
    for ( unsigned int i = 0; i < N; i++ )
    {
        printf( "vec[X,val][Y,val][Z,val][val,valRefInd]: X[ %.3f, %.3f ] Y[ %.3f, %.3f ] Z[ %.3f, %.3f ] val[ %.3f, %02i ]\n",
                float( pairFloatInd( vecXValInd[ i ] ).first ),
                float( vecValRef[ pairFloatInd( vecXValInd[ i ] ).second ] ),
                float( pairFloatInd( vecYValInd[ i ] ).first ),
                float( vecValRef[ pairFloatInd( vecYValInd[ i ] ).second ] ),
                float( pairFloatInd( vecZValInd[ i ] ).first ),
                float( vecValRef[ pairFloatInd( vecZValInd[ i ] ).second ] ),
                float( pairFloatInd( vecValInd[ i ] ).first ),
                int( pairFloatInd( vecValInd[ i ] ).second )
              );
    }
    
//========================================================================================================================================	
    //few device_vectors 2D iterator
    typedef device_vector< float >::iterator vecIter;
    device_vector< float > vec1D_1( 100, 0.1f );
    vec1D_1[ 2 ] = 17;    
    device_vector< float > vec1D_2( 100, 0.2f );
    device_vector< vecIter > vec2D( 2 );                //useful only for sort_by_key feature
    vec2D[ 0 ] = vec1D_1.begin();
    vec2D[ 1 ] = vec1D_2.begin();
    printf( "%.2f\n", float( vecIter(vec2D[ 0 ] )[ 2 ]) );
//=========================================================================================================================================    
    unsigned int colsX = 1920; unsigned int rowsY = 1060;
	vector< float > vecRowTmp( colsX, 1.0f );					
	vector< vector< float > > vecPicture( rowsY, vecRowTmp );		//example host luminance picture

//	typedef tuple< unsigned int, unsigned int, float > XYLumPix;    //thrust luminance pixs storage via tuple
//    device_vector < XYLumPix > GPUPicture( colsX * rowsY );                 //still problem of sending via PCIe => cudamemcpy2d; cudamemcpyArray
    /////vector< vector< Lum > > => LumArray	
	host_vector< float > hostPic( colsX * rowsY );
	for ( unsigned int rowY = 0; rowY < rowsY; rowY++ )
		std::copy( vecRowTmp.begin(), vecRowTmp.end(), hostPic.begin() );

	device_vector< float > gpuPic( colsX * rowsY );
	std::copy( vecRowTmp.begin(), vecRowTmp.end(), gpuPic.begin() );

    cudaDeviceSynchronize();
    return 0;
}

