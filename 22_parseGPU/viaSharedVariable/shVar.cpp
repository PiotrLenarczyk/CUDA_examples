/* http://advancedlinuxprogramming.com/ */
/* https://ubuntuforums.org/archive/index.php/t-1426536.html */
//STL
#include <iostream>
#include <sys/shm.h>
#include <sys/stat.h>

using namespace std;

struct MySharedData
{
    unsigned int size;
    float value[ (unsigned)17E6 ];
};

int getSHM( key_t key )
{
    int shmid = shmget( key, sizeof( struct MySharedData ), 0666 );
    if ( shmid < 0 )
    {
        perror( "shmget ERROR!" );
        return -1;
    }

    struct MySharedData* data = ( struct MySharedData* ) shmat( shmid, NULL, 0 );
    if ( ( long )data == -1 )
    {
        perror( "shmat ERROR!" );
        return -1;
    }

    data->value[ 2 ] = 15.1f;

    /* although not required, it is a good idea to detach */
    if ( shmdt( data ) < 0 )
    {
        perror( "shmdt ERROR!" );
        return -1;
    }
    return 0;
}

int main( void )
{
    key_t key = 1234;
    int shmid = shmget( key, sizeof( struct MySharedData ), IPC_CREAT | 0666 );

    if ( shmid < 0 )
    {
        perror( "shmget ERROR!" );
        return -1;
    }
    /* attach/map shared memory to our data type */
    struct MySharedData* data = ( struct MySharedData* )  shmat( shmid, NULL, 0 );
    cout << "data size: " << sizeof( struct MySharedData ) / ( 1024.0 * 1024.0 ) << "[MB]" << endl;
    if ( ( long )data == -1 )
    {
        perror( "shmat ERROR!" );
        return -1;
    }

    data->size = 3;
    data->value[ 0 ] = 123.56789;
    data->value[ 1 ] = 0.1f;
    data->value[ 2 ] = 0.2f;
    printf( "direct shared float [%i] data = %f\n", data->size, data->value[ 0 ] );

    getSHM( key );

//     while(1){}
//     return -1; //initialize memory and leavo for further usage - pass shmid to struct for destrying; quite useful rare instruction goto

    /* detach from shared memory */
    if ( shmdt( data ) < 0 )
    {
        perror( "shmdt ERROR!" );
        return -1;
    }

    /* destroy shared memory (important!!!) */
    if ( shmctl( shmid, IPC_RMID, NULL ) < 0 )
    {
        perror( "shmctl ERROR!" );
        return -1;
    }

    return 0;
}
