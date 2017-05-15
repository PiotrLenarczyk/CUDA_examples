/* 
 rm a.out aChild.out & make clean && qmake && make && /usr/local/cuda-8.0/bin/nvcc -arch=sm_50 shVarChild.cpp -o aChild.out && clear && ./a.out
 */
//STL
#include <iostream>
#include <stdio.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include "sharedStruct.h"
using std::cout; using std::endl; using std::cerr;

//HOST
long destroySHM;
int getSHM( key_t key ); //data->structShmid; data->size; data->value
void freeSHM( struct sharedData* data );
void closeSHM();
//GPU

int main( void )
{
    cout << endl << "========== GPU ================" << endl;
    getSHM( (key_t)1234 );
    
    /* the FPGA stuff here */
    
    freeSHM( data );
    cout << "========== END GPU ============" << endl << endl;
    return 0;
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
