//STL
#include <cstdlib>                  //atol
#include <iostream>                 //printf, cout, endl
#include <time.h>                   //timings
//CUDA
#include <thrust/device_vector.h>   //omit host vectors and costly data transfers
#include <thrust/sort.h>            //sorting
#include <thrust/random.h>          //random example data generator
#include <thrust/tuple.h>           //abstract data vectors
#include <thrust/iterator/zip_iterator.h>  //tuple iterator 

using namespace thrust;             //note that default is THRUST!!!
using std::cout; using std::endl;   //STL is for timings and data viewing

long int N( 0 );
    

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
    printf( "N = %ld\n", N );
    tuple< float, float, float, float > tuple4D( 10.1f, 75.0f, 11.1f, 1.1f ); //4D point tuple
    cout << "tuple4D[ 1 ]: " << get< 1 >( tuple4D ) << endl;
    
    //tuple pointer iterator - zip:
    default_random_engine rng( 123 );
    uniform_real_distribution< float > dist_f( 0.0f, 0.996f );
    device_vector< float > vecX( N ), vecY( N ), vecZ( N ), vecVal( N );
    for ( unsigned int i = 0; i < N; i++ )
    {
        vecX[ i ] = dist_f( rng );
        vecY[ i ] = dist_f( rng );
        vecZ[ i ] = dist_f( rng );
        vecVal[ i ] = dist_f( rng );
    }
    typedef device_vector< float >::iterator FloatIter;
    typedef tuple< FloatIter, FloatIter, FloatIter, FloatIter > tup4DIter;
    typedef zip_iterator< tup4DIter > zip4DIter;

    zip4DIter iter4D( make_tuple( 
                                    vecX.begin(),
                                    vecY.begin(),
                                    vecZ.begin(),
                                    vecVal.begin()
                                )
                    );
    cout << "zipIter4D[0]: " 
         << get< 0 >( iter4D[ 0 ] ) << ", "
         << get< 1 >( iter4D[ 0 ] ) << ", "
         << get< 2 >( iter4D[ 0 ] ) << ", "
         << get< 3 >( iter4D[ 0 ] ) << ", "
         << endl;    
    
    device_vector< pair< float, float > > mapTh( N ); //Thrust map structure
    for ( size_t i = 0; i < mapTh.size(); i++ )
    {
        float a = dist_f( rng );
        float b = dist_f( rng );
        mapTh[ i ] = make_pair( a, b );
    }
    cout << "unsorted map:" << endl;
    for ( int i = 0; i < mapTh.size(); i++ )
        printf( "map[ %02i ][ key ][ val ]: [ %.2f ][ %.2f ]\n", 
                i,
                pair< float, float >( mapTh[ i ] ).first,
                pair< float, float >( mapTh[ i ] ).second
        );
    cout << "thrust map are not default sorted!" << endl;
    
    t = clock();
    sort( mapTh.begin(), mapTh.end() );
    cout << "sorted map ";
    cout << "time: " << 1000 * float( clock() - t ) / CLOCKS_PER_SEC << "[ ms ]" << endl;
    for ( int i = 0; i < mapTh.size(); i++ )
        printf( "map[ %02i ][ key ][ val ]: [ %.2f ][ %.2f ]\n", 
                i,
                pair< float, float >( mapTh[ i ] ).first,
                pair< float, float >( mapTh[ i ] ).second
        );
    
    cout << "unsorted mmap:" << endl;
    typedef tuple< float, float, float, float > tup4D;
    tup4D tmpTup( 1 );
    device_vector< tuple< float, float, float, float > > map4DTh( N ); //Thrust multimap structure up to 10 datatypes
    for ( size_t i = 0; i < map4DTh.size(); i++ )
        map4DTh[ i ] = make_tuple( dist_f( rng ), dist_f( rng ), dist_f( rng ), dist_f( rng ) );
    
    for ( int i = 0; i < map4DTh.size(); i++ )
    {
        tmpTup = map4DTh[ i ];
        printf( "mmap[ %02i ][ key ][ vals ]: [ %.2f ][ %.2f, %.2f, %.2f ] \n", i, 
                get< 0 >( tmpTup ),
                get< 1 >( tmpTup ),
                get< 2 >( tmpTup ),
                get< 3 >( tmpTup ) );
    }
    
    t = clock();
    sort( map4DTh.begin(), map4DTh.end() );
    cout << "sorted mmap ";
    cout << "time: " << 1000 * float( clock() - t ) / CLOCKS_PER_SEC << " [ ms ]" << endl;
    for ( int i = 0; i < map4DTh.size(); i++ )
    {
        tmpTup = map4DTh[ i ];
        printf( "mmap[ %02i ][ key ][ vals ]: [ %.2f ][ %.2f, %.2f, %.2f ] \n", i, 
                get< 0 >( tmpTup ),
                get< 1 >( tmpTup ),
                get< 2 >( tmpTup ),
                get< 3 >( tmpTup ) );
    }
    
    cudaDeviceSynchronize();
    return 0;
}
