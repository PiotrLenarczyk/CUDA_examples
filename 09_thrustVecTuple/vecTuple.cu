//THRUST
#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
//STL
#include <iostream>
#include <vector>

int N = 10;
thrust::tuple< int, const char * > tString( N, "thrust" );

int main( void )
{
    std::cout << "The 1st value of tString is " << thrust::get< 0 >( tString ) << std::endl;
    std::vector < float > vecStd( N, 0.1f );
    thrust::host_vector< float > h_vec( N );
    h_vec = vecStd;
    std::cout << "h_vec from vecStd :" << std::endl;
    thrust::copy( h_vec.begin(), h_vec.end(), std::ostream_iterator< float >( std::cout, "\n" ) );
    
    thrust::device_vector< float > d_vec( N );
    d_vec = vecStd;
    std::cout << "d_vec from vecStd :" << std::endl;
    thrust::copy( d_vec.begin(), d_vec.end(), std::ostream_iterator< float >( std::cout, "\n" ) );
    return 0;
}
