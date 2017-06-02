#include "shMem.h"

using namespace std;

//HOST
unsigned i = 0;

int freeShMem( int &shmid )
{
    if ( shmctl( shmid, IPC_RMID, NULL ) < 0 )
    {
        cerr << "shmctl ERROR!\n";
        return -1;
    } 
    return 0;
}

//GPU
__device__ float d_arr1[ array1Size ];
__device__ float d_arr2[ array2Size ];
void freeGPU()
{  
    cudaFree( d_arr1 );
    cudaFree( d_arr2 );
    cudaDeviceSynchronize();
    cudaDeviceReset();
}

int main( void )
{
    int shmid = shmget( key, sizeof( Arrays ), 0666 ); if ( shmid < 0 ) { cerr << "shmget ERROR!\n"; return -1; }
    struct Arrays* someData = ( struct Arrays* )  shmat( shmid, NULL, 0 );
    if ( cudaMemcpyToSymbol( d_arr1, &someData->array1[ 0 ], sizeof( float ) * array1Size ) != cudaSuccess ) { cerr << "array1 GPU copy error!\n"; freeGPU(); return -1; }
    if ( cudaMemcpyToSymbol( d_arr2, &someData->array2[ 0 ], sizeof( float ) * array2Size ) != cudaSuccess ) { cerr << "array2 GPU copy error!\n"; freeGPU(); return -1; }
    
    
    /* destroy used shared memory (important!!!) */
    if ( freeShMem( shmid ) != 0 )  { cerr << "Shared Memory free error!\n"; freeGPU(); return -1; }
    freeGPU();
    return 0;
}
