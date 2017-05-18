#include <iostream>

using namespace std;

__device__ float4 a[ 10 ]; //a[].x; a[].y; a[].z; a[].w;

__global__ void print()
{
    float tmp;
    for ( unsigned i = 0; i < 10; i++ )
    {
        tmp = ( float )i;
        a[ i ] = make_float4( 1 * tmp, 2 * tmp, 3 * tmp, 4 * tmp );
        printf( "a[%i]: [ %f, %f, %f, %f ]\n", i, a[ i ].x, a[ i ].y, a[ i ].z, a[ i ].w );
    }
}

int main( void )
{
    print<<< 1, 1 >>>();
    
    cudaFree( a );
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}
