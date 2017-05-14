/* /usr/local/cuda-8.0/bin/nvcc -arch=sm_35 shVarChild.cpp -o aChild.out && ./aChild.out */
//STL
#include <iostream>
#include <stdio.h>
#include <sys/shm.h>
#include <sys/stat.h>

using std::cout; using std::endl; using std::cerr;

struct MySharedData
{
    unsigned int size;
    float value[ 30 ];
};

int getSHM( key_t key )
{
    cout << endl << endl << "============================" << endl;
    int shmid = shmget( key, sizeof( struct MySharedData ), 0666 );
    if ( shmid < 0 )
    {
        cerr << "shmget ERROR!";
        return -1;
    }

    struct MySharedData* data = ( struct MySharedData* ) shmat( shmid, NULL, 0 );
    if ( ( long )data == -1 )
    {
        cerr <<  "shmat ERROR!";
        return -1;
    }

    for ( unsigned i = 0; i < data->size; i++ )
        printf( "direct shared float [%i] data = %f\n", i, data->value[ i ] );
    
    /* although not required, it is a good idea to detach */
    if ( shmdt( data ) < 0 )
    {
        cerr << "shmdt ERROR!";
        return -1;
    }
    return 0;
}

int main( void )
{
    return getSHM( (key_t) 1234 );
}
