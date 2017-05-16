/* 
 rm a.out aChild.out & make clean && qmake && make && /usr/local/cuda-8.0/bin/nvcc -arch=sm_XX shVarChild.cu -o aChild.out && clear && ./a.out
 */
//STL
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <time.h>
#include "sharedStruct.h"
using std::cout; using std::endl; using std::cerr;

//HOST
unsigned i;
long destroySHM;
int getSHM( key_t key ); //data->structShmid; data->size; data->value
void freeSHM( struct sharedData* data );
void closeSHM();
//GPU
__device__ float d_pics[ picAllocX * picAllocY * pics ];
void freeGPU();

__global__ void printKernel()
{
    for ( unsigned i = 0; i < 3; i++ )
        printf( "d_pics[%i]: %f\n", i, d_pics[ i ] );
}

int main( void )
{
    cout << endl << "========== GPU ================" << endl;
    clock_t t = clock();
    getSHM( (key_t)1234 ); //load from shared memory to GPU memory
    if ( cudaMemcpyToSymbol( d_pics, &data->value[ 0 ], sizeof( float ) * data->size ) != cudaSuccess ) { cerr << "GPU copy error!\n"; freeGPU(); return -1; }
    printKernel<<< 1, 1 >>>();
    cudaDeviceSynchronize();
    cout << "CPU clocks ( data load to GPU ): " << float( clock() - t ) << endl;
    cout << "allocated GPU memory usage: " << 100 * (float)data->size / (float)(picAllocX * picAllocY * pics) << "[%] for: " << data->picsNo << " from allocated " << pics << " pictures" << endl;
    freeGPU();
    freeSHM( data );
    cout << "========== END GPU ============" << endl << endl;
    return 0;
}

void freeGPU()
{
    cudaFree( d_pics );
    cudaDeviceSynchronize();
    cudaDeviceReset();
}

void freeSHM( struct sharedData* data )
{
    if ( shmdt( data ) < 0 )
        cerr << "child shmdt ERROR!\n";
}

void closeSHM( long destroySHM )
{
    if ( shmctl( destroySHM, IPC_RMID, NULL ) < 0 )
        cerr << "child shmctl ERROR!\n";
}

int getSHM( key_t key )
{
    int shmid = shmget( key, sizeof( struct sharedData ), 0666 );
    if ( shmid < 0 )
    {
        cerr << "child shmget ERROR!\n";
        return -1;
    }

    data = ( struct sharedData* ) shmat( shmid, NULL, 0 );
    if ( ( long )data == -1 )
    {
        cerr <<  "child shmat ERROR!\n";
        return -1;
    }
    destroySHM = data->structShmid;
    return 0;
}
