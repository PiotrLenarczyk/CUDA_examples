#include <iostream>
#include <chrono>

using namespace std;

//CPU
typedef unsigned int uint;
int i = 0;
int gpuCount = 0;
//GPU
cudaDeviceProp gpuProperties;
const uint N = 1E8;
const uint unrolling = 32;
__global__ void loop();
__global__ void unrollLoop();

//auto t1 = std::chrono::high_resolution_clock::now(); //highest possible standard chronometrics
//cout << "int took " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() << endl;
void initalizeHost( float * const ip, int const  size )
{
    for ( size_t i = 0; i < size; i++ )
        ip[i] = 0.0f;
}

int main( void )
{
    cudaGetDeviceCount( &gpuCount );
    //HOST
    float *h_arr[ gpuCount ];
    uint perDevN = 1E8 / gpuCount;
    //DEVICE
    cudaStream_t stream[ gpuCount ];
    float *d_arr[ gpuCount ];
    //alocate H,D memories
    for ( i = 0; i < gpuCount; i++ )
    {
        //HOST 
        cudaMallocHost( ( void** ) &h_arr[ i ], perDevN );
        //DEVICE
        cudaSetDevice( i );
        cudaGetDeviceProperties( &gpuProperties, i );
        cout << gpuProperties.name << ": " << endl;
        cudaMalloc( ( void** ) &d_arr, perDevN );
        cudaStreamCreate( &stream[ i ] );
    }
    //initalize data
    for ( i = 0; i < gpuCount; i++ )
    {
        initalizeHost( h_arr[ i ], perDevN );
        cudaSetDevice( i );
        
    }
    
    //free memory
    for ( i = 0; i < gpuCount; i++ )
    {
        //HOST 
        cudaFreeHost( h_arr[ i ] );
        //DEVICE
        cudaSetDevice( i );
        cudaFree( d_arr[ i ] );
        cudaStreamDestroy( stream[ i ] );
    }

    cudaDeviceReset();
    return 0;
}
