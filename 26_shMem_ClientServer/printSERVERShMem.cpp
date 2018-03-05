//STL
#include "shMem.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

using namespace std;

unsigned shmidPrint = 0;
char destroyFlag[ 2 ] = { '-','d' };
int main ( int argc, char *argv[] )
{
    if ( argc <= 1 ) { cerr << "ERROR too few arguments!"; return -1; }
    shmidPrint = ( unsigned )strtoull( argv[ 1 ], NULL, 0 );
    if ( ( argc == 3 ) && ( strcmp( destroyFlag, argv[ 2 ] ) == 0 ) )
    { 
        cout << "destroy shared memory" << endl; 
        if ( shmctl( shmidPrint, IPC_RMID, NULL ) < 0 ) cerr << "child shmctl ERROR!\n"; 
    }
    struct Arrays* ArrPrint = ( struct Arrays* ) shmat( shmidPrint, NULL, 0 );
    if ( ( long )ArrPrint == -1 ) { cerr <<  "child shmat ERROR!\n"; return -1; }
    
    cout << "ArrPrint->isBeingWritten: [" << unsigned( ArrPrint->isBeingWritten ) << "]" << endl;
    cout << "ArrPrint->shmid: [" << ArrPrint->shmid << "]" << endl;
    for ( unsigned i = 0; i < array1Size; i++ )
        cout << "ArrPrint->array1[ " << i << " ]: [" << ArrPrint->array1[ i ] << "]" << endl;
    for ( unsigned i = 0; i < array2Size; i++ )
        cout << "ArrPrint->array2[ " << i << " ]: [" << ArrPrint->array2[ i ] << "]" << endl;

    return 0;
}
