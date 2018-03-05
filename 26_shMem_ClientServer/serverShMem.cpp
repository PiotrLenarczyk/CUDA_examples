#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h> 
#include <sys/socket.h>
#include <netinet/in.h>
#include "shMem.h"

using namespace std;

unsigned i = 0;
void receiveStruct( struct Arrays *ArrayIn, int argc, char *argv[] );
unsigned serverShmid = 0;
struct Arrays* shMemObtain();
int destroyShMem( int shmid );  /* destroy used shared memory (important!!!) */
void error( const char *msg );


int main( int argc, char *argv[] )
{    
//=============================================
            struct Arrays* someData = shMemObtain();
            receiveStruct( someData, argc, argv );
//          if all local client-operations where performed destroy shared memory
            if ( destroyShMem( serverShmid ) != 0 ) { cerr << "shared memory destroy problem!\n"; return -1; }
//=============================================

    return 0;
}

void receiveStruct( struct Arrays *ArrayIn, int argc, char *argv[] )
{
    
//======= INITALIZE MULTIPLE CLIENTS SERVER
    int sockfd, newsockfd, portno, pid;
    socklen_t clilen;
    struct sockaddr_in serv_addr, cli_addr;

    if ( argc < 2 ) {
        fprintf( stderr,"ERROR, no port provided\n" );
        exit( 1 );
    }
    sockfd = socket( AF_INET, SOCK_STREAM, 0 );
    if ( sockfd < 0 ) 
        error( "ERROR opening socket" );
    bzero( ( char * ) &serv_addr, sizeof( serv_addr ) );
    portno = atoi( argv[ 1 ] );
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons( portno ); 
    if ( bind( sockfd, ( struct sockaddr * ) &serv_addr,
            sizeof( serv_addr ) ) < 0 ) 
            {   error( "ERROR on binding" ); close( newsockfd ); close( sockfd ); }
    cout << "listening on port: [" << portno << "] to SERVER shared memory shmid: [" << serverShmid << "]" << endl;
    listen( sockfd, 5 );
    clilen = sizeof( cli_addr );
    while ( 1 ) 
    {
        newsockfd = accept( sockfd, 
            ( struct sockaddr * ) &cli_addr, &clilen );
        if ( newsockfd < 0 ) 
            error( "ERROR on accept" );
        pid = fork();
        if ( pid < 0 )
            error( "ERROR on fork" );
        if ( pid == 0 )  
        {
            close( sockfd );
//=============================================
        unsigned structSize = sizeof( *ArrayIn );
        unsigned char buffStruct[ structSize ];
        int n = read( newsockfd, buffStruct, structSize );   
        if ( n < 0 )
            error( "ERROR reading from socket" );
        cout << "bufferIn: [";
        for ( unsigned j = 0; j < structSize; j++ )
            cout << buffStruct[ j ];
        cout << "]" << endl;
        
        memcpy( ArrayIn, buffStruct, structSize );
        cout << "ArrayIn->isBeingWritten: [" << unsigned( ArrayIn->isBeingWritten ) << "]" << endl;
        cout << "ArrayIn->shmid: [" << ArrayIn->shmid << "]" << endl;
        for ( unsigned i = 0; i < array1Size; i++ )
            cout << "ArrayIn->array1[ " << i << " ]: [" << ArrayIn->array1[ i ] << "]" << endl;
        for ( unsigned i = 0; i < array2Size; i++ )
            cout << "ArrayIn->array2[ " << i << " ]: [" << ArrayIn->array2[ i ] << "]" << endl;
        n = write( newsockfd, "I got your message", 18 );
        if ( n < 0 ) 
            error( "ERROR writing to socket" );
//=============================================
            exit( 0 );
        }
        else close( newsockfd );
    } /* end of while */
    close( sockfd ); 
    

}

void error( const char *msg )
{
    perror( msg );
    exit( 1 );
}

struct Arrays* shMemObtain()
{
//    ======== RESERVE SHARED MEMORY
	/* obtain shared memory container */
    int shmid = shmget( key, sizeof( Arrays ), IPC_CREAT | 0666 ); if ( shmid < 0 ) 
		{ cerr << "shmget ERROR!\n"; return NULL; }
    /* attach/map shared memory to our data type */
    struct Arrays* someData = ( struct Arrays* )  shmat( shmid, NULL, 0 );
    serverShmid = shmid;    //current shared memory identification number obtained by key_t key
	someData->isBeingWritten = 0;
    for ( unsigned i = 0; i < array1Size; i++ )
        someData->array1[ i ] = i + 0.1f;
    for ( unsigned i = 0; i < array2Size; i++ )
        someData->array2[ i ] = i + 0.23f;
    return someData;
}

int destroyShMem( int shmid )  /* destroy used shared memory (important!!!) */
{
    if ( shmctl( shmid, IPC_RMID, NULL ) < 0 )
    {
        cerr << "shmctl ERROR!\n";
        return -1;
    }
    return 0;
}
