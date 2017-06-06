#include <stdlib.h>
#include <iostream>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

using namespace std;

//================ HOST ========================
const unsigned no = 4;
float h_a;
typedef struct d_fpMedian
{
    float f[ no ];
    float med[ 1 ];
}d_fpMedian;

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
__global__ void printStruct()
{
    unsigned ind = threadIdx.x;
    printf( "d_fpm[ 0 ].f[ %i ]: %.3f\n", ind, d_fpm[ 0 ].f[ ind ] );
}

int main( void )
{
    print<<< 1, 1 >>>(); 
//========= DEVICE STRUCT ALLOCATION
    struct d_fpMedian *d_mall; 
    if (  cudaMalloc( &d_mall, sizeof( d_fpMedian ) )  != cudaSuccess ) { cerr << "GPU struct allocation error!\n"; return -1; }
    if ( cudaMemcpyToSymbol( d_fpm, &d_mall, sizeof( d_fpMedian* ) ) != cudaSuccess ) { cerr << "GPU struct Ptr allocation error!\n"; return -1; }
//==================================
    float h_f[ no ];
    for ( unsigned i = 0; i < no; i++ )
        if ( i%2 )
            h_f[ i ] = float( i ) + 0.1f;
        else
            h_f[ i ] = -float( i ) + 0.2f;
    for ( unsigned i = 0; i < no; i++ )
        cout << "h_f[" << i << "]: " << h_f[ i ] << endl;
    if ( cudaMemcpyToSymbolAsync( d_f, &h_f, sizeof( h_f ) ) != cudaSuccess ) { cerr << "GPU H2D copy error!\n"; return -1; }
    populateStruct<<< 1, no >>>();
    printStruct<<< 1, no >>>();
    
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
    printf( "===========\ndevice struct processing:\n===========\n" );
}

__global__ void populateStruct()
{
    unsigned ind = threadIdx.x;
    if ( ind == 0 ) d_fpm[ 0 ].med[ 0 ] = 0.0f;
    d_fpm[ 0 ].f[ ind ] = d_f[ ind ];
    printf( "d_fpm[ 0 ].f[ %i ]: %.3f\n", ind, d_fpm[ 0 ].f[ ind ] );
    __syncthreads();
    if ( ind == 0 )
        recalcMedian<<< 1, no >>>( d_fpm ); //struct median normalization - keep float data in normalized range approx.: [ -1, +1 ]
    __syncthreads();
}

__global__ void recalcMedian( struct d_fpMedian *d_fpm ) //only on device callable!
{
    unsigned ind = threadIdx.x;
    d_f[ ind ] = d_fpm[ 0 ].f[ ind ];
    __syncthreads();
    if ( ind == 0 )
        thrust::sort( thrust::device, d_f, d_f + no );
    __syncthreads();
    float median = d_f[ no / 2 ];
    if ( ind == 0 ) printf( "sort on temporary array: \n" );
    printf( "d_f[%i]: %f\n", ind, d_f[ ind ] );
    d_fpm[ 0 ].med[ 0 ] += median;
    if ( ind == 0 ) printf( "recalculated struct with its median value: %f\n", median );
    d_fpm[ 0 ].f[ ind ] = d_fpm[ 0 ].f[ ind ] - median;    
}
