#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h> 
#include <iostream>
#include "shMem.h"

using namespace std;

struct Arrays* shMemClient();
int destroyShMem( int shmid );
void sendStruct( struct Arrays &Arr, int argc, char *argv[] );
void error( const char *msg );

//GPU variables:
__device__ float d_array1[ array1Size ];
__device__ float d_array2[ array2Size ];
void freeGPU()
{
    cudaFree( d_array1 );
    cudaFree( d_array2 );
    cudaDeviceSynchronize();
    cudaDeviceReset();
}

int main( int argc, char *argv[] )
{   cudaDeviceReset();
    //=============================================    
    struct Arrays* someData = shMemClient();
//  compute stuff on client GPU...
//  ...
//  ...
    //=============================================    
    sendStruct( *someData, argc, argv );    //send results on server via TCP
//  if all local client-operations where performed destroy shared memory
    if ( destroyShMem( someData->shmid ) != 0 ) { cerr << "shared memory destroy problem!\n"; return -1; }
    
    freeGPU();
    return 0;
}


void error( const char *msg )
{
    perror( msg );
    exit( 0 );
}

struct Arrays* shMemClient()
{
//    ======== RESERVE SHARED MEMORY
	/* obtain shared memory container */
    int shmid = shmget( key, sizeof( Arrays ), IPC_CREAT | 0666 ); if ( shmid < 0 ) 
		{ cerr << "shmget ERROR!\n"; return NULL; }
    /* attach/map shared memory to our data type */
    struct Arrays* someData = ( struct Arrays* )  shmat( shmid, NULL, 0 );
    someData->shmid = shmid;    //current shared memory identification number obtained by key_t key
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

void sendStruct( struct Arrays &Arr, int argc, char *argv[] )
{
       
    int sockfd, portno, n;
    struct sockaddr_in serv_addr;
    struct hostent *server;

    if ( argc < 3 ) 
    {
       fprintf( stderr, "usage %s hostname port\n", argv[ 0 ] );
       exit( 0 );
    }
    portno = atoi( argv[ 2 ] );
    sockfd = socket( AF_INET, SOCK_STREAM, 0 );
    if ( sockfd < 0 ) 
        error( "ERROR opening socket" );
    server = gethostbyname( argv[ 1 ] );
    if ( server == NULL ) 
    {
        fprintf( stderr, "ERROR, no such host\n" );
        exit( 0 );
    }
    bzero( ( char * )&serv_addr, sizeof( serv_addr ) );
    serv_addr.sin_family = AF_INET;
    bcopy( ( char * )server->h_addr, 
         ( char * )&serv_addr.sin_addr.s_addr,
         server->h_length );
    serv_addr.sin_port = htons( portno );
    if ( connect( sockfd, ( struct sockaddr * )  &serv_addr, sizeof( serv_addr ) ) < 0 )
        error( "ERROR connecting" );
    
//=============================================    
    for ( unsigned i = 0; i < array1Size; i++ )
        cout << "sent array1[" << i << "]: " << Arr.array1[ i ] << endl;
    for ( unsigned i = 0; i < array2Size; i++ )
        cout << "sent array2[" << i << "]: " << Arr.array2[ i ] << endl;
    unsigned chStructBuffSize =  sizeof( Arr );
    unsigned char chStructBuff[ chStructBuffSize ];
    memcpy( chStructBuff, ( unsigned char* )( &Arr ), chStructBuffSize );
    cout << "bufferOut: [";
    for ( unsigned i = 0; i < chStructBuffSize; i++ )
         cout << chStructBuff[ i ];
    cout << "]" << endl;
    n = write( sockfd, chStructBuff, chStructBuffSize );
    if ( n < 0 ) 
         error( "ERROR writing to socket" );
//=============================================    
//=============================================    

    char bufferOut[ 255 ];
    bzero( bufferOut, 255 );
    n = read( sockfd, bufferOut, 255 );
    if ( n < 0 )
         error( "ERROR reading from socket" );
    printf( "%s\n", bufferOut );
    close( sockfd );
}
