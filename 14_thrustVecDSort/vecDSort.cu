//STL
#include <cstdlib>                  //atol
#include <iostream>                 //printf, cout, endl
#include <time.h>                   //timings
#include <vector>					//vector for example host picture
//CUDA
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>   //omit host vectors and costly data transfers
#include <thrust/sort.h>            //sorting
#include <thrust/random.h>          //random example data generator
#include <thrust/tuple.h>           //abstract data vectors
#include <thrust/sequence.h>        //data input index sequence

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
    cout << "sorting time: " << 1000 * float( clock() - t ) / CLOCKS_PER_SEC << "[ ms ]" << endl;
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
    vector< float > vecRowTmp( colsX, 1.01f );					
    vector< vector< float > > vecPicture( rowsY, vecRowTmp );		//example host luminance picture
    
    t = clock();
    device_vector< float > gpuPic( colsX * rowsY, 0.0f );
    for ( unsigned int rowY = 0; rowY < rowsY; rowY++ )
            thrust::copy( vecPicture[ rowY ].begin(), vecPicture[ rowY ].end(), gpuPic.begin() + ( rowY * colsX ) );
    cout << "HtD pic transfer time: " << 1000 * float( clock() - t ) / CLOCKS_PER_SEC << "[ ms ]" << endl;
    cout << "vector< vector< Lum > > => LumArray; pixs are accessible via colX, rowY already!" << endl;
    
    device_vector< unsigned int > ind1DPic ( colsX * rowsY );
    sequence( ind1DPic.begin(), ind1DPic.end(), 0.0f );
    cout << "ind1DPic[4]: " << ind1DPic[ 4 ] << endl;
    
    cout << "generatig < floatVal, unsigned intInd > pairs in parallel:" << endl;
    t = clock();
    auto valIndBeg = thrust::make_zip_iterator( thrust::make_tuple( gpuPic.begin(), ind1DPic.begin() ) );
    auto valIndEnd = thrust::make_zip_iterator( thrust::make_tuple( gpuPic.end(), ind1DPic.end() ) );
    sort( valIndBeg, valIndEnd );
    cout << "gpuPic parallel sorting time: " << 1000 * float( clock() - t ) / CLOCKS_PER_SEC << "[ ms ]" << endl;

    t = clock();
    typedef pair< float, unsigned int > pairFloatIndGPUInd; //< float, index > pair
    device_vector< pairFloatIndGPUInd > vecPicValInd( colsX * rowsY );
    for ( unsigned int i = 0; i < colsX * rowsY; i++ )
        vecPicValInd[ i ] = make_pair( gpuPic[ i ], ind1DPic[ i ] );
    sort( vecPicValInd.begin(), vecPicValInd.end() );
    cout << "<floatVal, uintInd> gpuPicSequentiallyGeneratdPairs sorting time: " << 1000 * float( clock() - t ) / CLOCKS_PER_SEC << "[ ms ]" << endl;
    
    cudaDeviceSynchronize();
    return 0;
}

