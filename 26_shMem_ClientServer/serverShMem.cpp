#include "shMem.h"
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h> 
#include <sys/socket.h>
#include <netinet/in.h>

using namespace std;

unsigned i = 0;

void someFunction( int );
void error(const char *msg)
{
    perror(msg);
    exit(1);
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

    if (argc < 2) {
        fprintf(stderr,"ERROR, no port provided\n");
        exit(1);
    }
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) 
        error("ERROR opening socket");
    bzero((char *) &serv_addr, sizeof(serv_addr));
    portno = atoi(argv[1]);
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(portno);
    if (bind(sockfd, (struct sockaddr *) &serv_addr,
            sizeof(serv_addr)) < 0) 
            error("ERROR on binding");
    listen(sockfd,5);
    clilen = sizeof(cli_addr);
    while (1) {
        newsockfd = accept(sockfd, 
            (struct sockaddr *) &cli_addr, &clilen);
        if (newsockfd < 0) 
            error("ERROR on accept");
        pid = fork();
        if (pid < 0)
            error("ERROR on fork");
        if (pid == 0)  {
            close(sockfd);
            someFunction(newsockfd);
            exit(0);
        }
        else close(newsockfd);
    } /* end of while */
    close(sockfd);
        
    return 0;
}

// void someFunction (int sock)
// {
//     int n;
//     float floatsIn[ 4 ];
//     for ( unsigned i = 0; i < 4; i++ )
//         floatsIn[ i ] = 0.0f;
//     n = read( sock, floatsIn, 3 );
//     if ( n < 0 ) error( "ERROR reading from socket" );
//     for ( unsigned i = 0; i < 4; i++ )
//         printf( "Here is the float[ %i ]: %f\n", i, floatsIn[ i ] );
//     n = write( sock, "I got your message", 18 );
//     if ( n < 0 ) error( "ERROR writing to socket" );
// }


void someFunction (int sock)
{
    int n; unsigned no = 4;
    unsigned buffSize = no * sizeof( float );
    unsigned char buffer[ buffSize ];
        //datatype, length, data[]
    for ( unsigned i = 0; i < buffSize; i++ )
        buffer[ i ] = ' ';
    n = read( sock, buffer, buffSize - 1 );   if ( n < 0 ) error( "ERROR reading from socket" );
    cout << "bufferIn: [";
    for ( unsigned j = 0; j < buffSize; j++ )
        cout << buffer[ j ];
    cout << "]" << endl;
    float floatsIn[ no ];
    for ( unsigned i = 0; i < no; i++ )
    {
        memcpy( ( unsigned char* )( &floatsIn[ i ] ),  &buffer[ i * sizeof( float ) ], sizeof( float ) );
    }
    for ( unsigned i = 0; i < no; i++ )
        printf( "floatsIn[ %i ]: %f\n", i, floatsIn[ i ] );
    n = write( sock, "I got your message", 18 );
    if ( n < 0 ) error( "ERROR writing to socket" );
}

// void someFunction (int sock)
// {
//    int n;
//    char buffer[4];
//       //datatype, length, data[]
//    bzero(buffer,4);
//    n = read(sock,buffer,3);
//    if (n < 0) error("ERROR reading from socket");
//     printf("Here is the float[0]: %f\n",atof(buffer));
//    n = write(sock,"I got your message",18);
//    if (n < 0) error("ERROR writing to socket");
// }

