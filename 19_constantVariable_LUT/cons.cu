//  http://cuda-programming.blogspot.com/2013/01/what-is-constant-memory-in-cuda.html
//STL
#include <iostream>

__constant__ float d_angle[ 360 ]; //constant memory LUT candidate

__global__ void test_kernel( float* d_array );

int main( int argc, char** argv )
{
    unsigned size = 3200;
    float* d_array;
    float h_angle[ 360 ];

    //allocate device memory
    cudaMalloc( ( void** ) &d_array, size * sizeof( float ) );

    //initialize allocated memory
    cudaMemset( d_array, 0, sizeof( float ) * size );

    //initialize angle array on host
    for( unsigned loop=0; loop < 360; loop++ )
        h_angle[ loop ] = acos( -1.0f ) * loop/ 180.0f;

    //copy host angle data to constant memory
    cudaMemcpyToSymbol( d_angle, h_angle, 360 * sizeof( float ) );
   
    test_kernel<<< size / 64, 64>>>( d_array );

    //constant variable view
    float DtH_angle[ 360 ]; cudaMemcpyFromSymbol( DtH_angle, d_angle, 360 * sizeof( float ) );
        for ( unsigned i = 0; i < 360; i++ )
    {
        printf( "[ind=%02i]: h_angle: [ %.2f ]; DtH_angle: [ %.2f ] \n",
                i,
                h_angle[ i ],
                DtH_angle[ i ]
              );
    }
    
    //free device memory
    cudaFree( d_array );
    cudaFree( d_angle );
    return 0;
}

__global__ void test_kernel(float* d_array)
{
    //calculate each thread global index
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    #pragma unroll 10
    for( unsigned loop=0; loop < 360; loop++ )
        d_array[ index ] = d_array[ index ] + d_angle[ loop ];
    return;
}
