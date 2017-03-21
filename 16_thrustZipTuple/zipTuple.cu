//http://stackoverflow.com/questions/36436432/cuda-thrust-zip-iterator-tuple-transform-reduce

//STL
#include <iostream>
#include <stdlib.h>
//Thrust
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/zip_iterator.h>

using std::cout; using std::endl;

typedef thrust::device_vector< float > dvec;
typedef thrust::tuple< float, float > tup;

struct func
{
  __device__ float operator()( tup t ) //difsq
  {
     float f = thrust::get< 0 >( t ) - thrust::get< 1 >( t );
     return f*f;
  }
};

int main()
{
  dvec a( 4, 4.f );
  dvec b( 4, 2.f );
  auto begin = thrust::make_zip_iterator( thrust::make_tuple( a.begin(), b.begin() ) );
  auto end = thrust::make_zip_iterator( thrust::make_tuple( a.end(), b.end() ) );
  cout << thrust::transform_reduce( begin, end, func(), 0.0f, thrust::plus< float >() ) << endl;
  cout << "done" << endl;
  
  return 0;
}
