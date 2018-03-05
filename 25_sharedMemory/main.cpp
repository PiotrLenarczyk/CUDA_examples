#include "shMem.h"

using namespace std;

unsigned i = 0;

int main( void )
{    
	/* obtain shared memory container */
    int shmid = shmget( key, sizeof( Arrays ), IPC_CREAT | 0666 ); if ( shmid < 0 ) 
		{ cerr << "shmget ERROR!\n"; return -1; }
    /* attach/map shared memory to our data type */
    struct Arrays* someData = ( struct Arrays* )  shmat( shmid, NULL, 0 );
    someData->shmid = shmid;
	someData->isBeingWritten = 0;
    for ( i = 0; i < array1Size; i++ )
        someData->array1[ i ] = i + 0.1f;
    for ( i = 0; i < array2Size; i++ )
        someData->array2[ i ] = i + 0.23f;
    //there must be a client program!
    
    return 0;
}

