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

void receiveFloat( int &sock, unsigned &no );
void receiveStruct( int &sock );
void error( const char *msg )
{
    perror( msg );
    exit( 1 );
}

int main( int argc, char *argv[] )
{    
//======== RESERVE SHARED MEMORY
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
            error( "ERROR on binding" );
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
//             unsigned noFloats = 4; receiveFloat( newsockfd, noFloats );
            receiveStruct( newsockfd );
            exit( 0 );
        }
        else close( newsockfd );
    } /* end of while */
    close( sockfd ); 
        
    return 0;
}

void receiveStruct( int &sock )
{
    Arrays ArrIn;
    unsigned buffSize = sizeof( ArrIn );    //+1 bigger
    unsigned char buffStruct[ buffSize ];
    int n = read( sock, buffStruct, buffSize );   
    if ( n < 0 )
        error( "ERROR reading from socket" );
    cout << "bufferIn: [";
    for ( unsigned j = 0; j < sizeof( ArrIn ); j++ )
        cout << buffStruct[ j ];
    cout << "]" << endl;
    
    memcpy( &ArrIn, buffStruct, sizeof( ArrIn ) );
    cout << "ArrIn.isBeingWritten: [" << unsigned( ArrIn.isBeingWritten ) << "]" << endl;
    cout << "ArrIn.shmid: [" << ArrIn.shmid << "]" << endl;
    for ( unsigned i = 0; i < array1Size; i++ )
        cout << "ArrIn.array1[ " << i << " ]: [" << ArrIn.array1[ i ] << "]" << endl;
    for ( unsigned i = 0; i < array2Size; i++ )
        cout << "ArrIn.array2[ " << i << " ]: [" << ArrIn.array2[ i ] << "]" << endl;
    n = write( sock, "I got your message", 18 );
    if ( n < 0 ) 
        error( "ERROR writing to socket" );
}

void receiveFloat( int &sock, unsigned &no )
{
    int n;
    unsigned buffSize = ( no + 1 ) * sizeof( float );
    unsigned char bufferFloat[ buffSize ];
    for ( unsigned i = 0; i < buffSize; i++ )
        bufferFloat[ i ] = ' ';
    n = read( sock, bufferFloat, buffSize - 1 );   
    if ( n < 0 ) 
        error( "ERROR reading from socket" );
    cout << "bufferFloatsIn: [";
    for ( unsigned j = 0; j < buffSize; j++ )
        cout << bufferFloat[ j ];
    cout << "]" << endl;
    float floatsIn[ no ];
    for ( unsigned i = 0; i < no; i++ )
    {
        memcpy( ( unsigned char* )( &floatsIn[ i ] ),  &bufferFloat[ i * sizeof( float ) ], sizeof( float ) );
    }
    for ( unsigned i = 0; i < no; i++ )
        printf( "floatsIn[ %i ]: %f\n", i, floatsIn[ i ] );
    n = write( sock, "I got your message", 18 );
    if ( n < 0 ) error( "ERROR writing to socket" );
}


