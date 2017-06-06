#include <iostream>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

using namespace std;

//================ HOST ========================
const unsigned no = 4;
float h_a;
struct d_fpMedian
{
    float f[ no ];
    float f_tmp[ no ];
    float med[ 1 ];
};

//================ GPU ========================
__device__ float4 d_a[ no ]; //d_a[].x; d_a[].y; d_a[].z; d_a[].w;
__device__ float d_f[ no ];
__device__ struct d_fpMedian *d_fpm;
void freeGPU()
{
    cudaFree( d_f );
    cudaFree( d_fpm );
    cudaFree( d_a );
}
//GPU functions
__global__ void populateStruct();
__global__ void recalcMedian( struct d_fpMedian *d_fpm ); //only on device callable!
__global__ void print();

int main( void )
{
    float h_f[ no ];
    for ( unsigned i = 0; i < no; i++ )
        if ( i%2 )
            h_f[ i ] = i + 0.1f;
        else
            h_f[ i ] = -i + 0.2f;
    cudaMemcpyToSymbolAsync( d_f, &h_f, no );
    print<<< 1, 1 >>>();
    populateStruct<<< 1, no >>>();
    
    freeGPU();
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}


__global__ void print()
{
    float i;
    for ( i = 0; i < no; i++ )
    {
        d_a[ ( unsigned )i ] = make_float4( 1 * i, 2 * i, 3 * i, 4 * i );
        printf( "d_a[%i]: [ %f, %f, %f, %f ]\n", i, d_a[ ( unsigned )i ].x, d_a[ ( unsigned )i ].y, d_a[ ( unsigned )i ].z, d_a[ ( unsigned )i ].w );
    }
}

__global__ void populateStruct()
{
    unsigned ind = threadIdx.x;
    d_fpm->f[ ind ] = d_f[ ind ];
    syncthreads();
    if ( ind == 0 )
        recalcMedian<<< 1, no >>>( d_fpm ); //struct median normalization - keep float data in normalized range approx.: [ -1, +1 ]
}

__global__ void recalcMedian( struct d_fpMedian *d_fpm ) //only on device callable!
{
    unsigned ind = threadIdx.x;
    d_fpm->f_tmp[ ind ] = d_fpm->f[ ind ];
    syncthreads();
    if ( ind == 0 )
        thrust::sort( thrust::device, d_fpm->f_tmp, d_fpm->f_tmp + no );
    syncthreads();
    float median = d_fpm->f_tmp[ no / 2 ];
    d_fpm->f[ ind ] /= median;
    
}
