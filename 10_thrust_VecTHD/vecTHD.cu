//THRUST
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
//STL
#include <iostream>
#include <iomanip>

// define a 4d float vector
typedef thrust::tuple< float, float, float, float > vec4;

// return a random vec4 in [0,1)^2
vec4 make_random_vec4( void )
{
  static thrust::default_random_engine rng;
  static thrust::uniform_real_distribution< float > u01( 0.0f, 1.0f );
  float x1D = u01( rng );
  float y2D = u01( rng );
  float z3D = u01( rng );
  float x4D = u01( rng );
  return vec4( x1D, y2D, z3D, x4D );
}

int main ( void )
{
    const size_t N = 1000000;
    // allocate some random points on the host
    thrust::host_vector<vec4> h_5Dpoints( N );
    thrust::generate( h_5Dpoints.begin(), h_5Dpoints.end(), make_random_vec4 );
    std::cout << "The x4D[ 0 ] of h_5Dpoints is " << thrust::get< 3 >( h_5Dpoints[0] ) << std::endl;
    
    // transfer to device
    thrust::device_vector< vec4 > d_5Dpoints = h_5Dpoints;
    vec4 p = d_5Dpoints[ 0 ]; std::cout << "The x4D[ 0 ] of d_5Dpoints is " << thrust::get< 3 >( p ) << std::endl;
    std::cout << "The x4D[ 0 ] of d_5Dpoints is " << thrust::get< 3 >( vec4( d_5Dpoints[ 0 ] ) ) << std::endl;
    
    return 0;
}
