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

void sendStruct( struct Arrays &Arr, int &n, int &sockfd );
void sendFloat( float *floatsOut, int &n, int &sockfd );

void error( const char *msg )
{
    perror( msg );
    exit( 0 );
}

int main( int argc, char *argv[] )
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

//     float floatsOut[ 4 ];
//     for ( unsigned i = 0; i < 4; i++ )
//         floatsOut[ i ] = i + 0.111111f;
//     sendFloat( floatsOut, n, sockfd );
    
    Arrays ArrOut;
    for ( unsigned i = 0; i < array1Size; i ++ )
    {
        ArrOut.isBeingWritten = 0;
        ArrOut.shmid = 15;
        ArrOut.array1[ i ] = 0.1f + i;
    }
    for ( unsigned i = 0; i < array2Size; i ++ )
    {
        ArrOut.array2[ i ] = 0.2f + i;
    }
    sendStruct( ArrOut, n, sockfd );
    
    
    char bufferOut[ 255 ];
    bzero( bufferOut, 255 );
    n = read( sockfd, bufferOut, 255 );
    if ( n < 0 )
         error( "ERROR reading from socket" );
    printf( "%s\n", bufferOut );
    close( sockfd );
    return 0;
}

void sendStruct( struct Arrays &Arr, int &n, int &sockfd )
{
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
}

void sendFloat( float *floatsOut, int &n, int &sockfd )
{
    for ( unsigned i = 1; i < 4; i++ )
    {
        cout << "floatsOut[" << i <<"]: " << floatsOut[ i ] << endl;
    }
    for ( unsigned i = 0 ;i < 4; i++ )
        printf( "entered [%i]: %f\n ", i, floatsOut[ i ] );
    unsigned chBuffSize =  4 * sizeof( float );
    unsigned char chFloats[ chBuffSize ];
    for ( unsigned i = 0; i < 4; i++ )
        memcpy( &chFloats[ i * sizeof( float ) ], ( unsigned char* )( &floatsOut[ i ] ), sizeof( float ) );
    cout << "bufferOut: [";
    for ( unsigned i = 0; i < chBuffSize; i++ )
         cout << chFloats[ i ];
    cout << "]" << endl;
    n = write( sockfd, chFloats, chBuffSize );
    if ( n < 0 )
        error( "ERROR writing to socket" );
}
