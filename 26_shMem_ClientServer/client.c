#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h> 
#include <iostream>

using namespace std;
void error(const char *msg)
{
    perror(msg);
    exit(0);
}

int main(int argc, char *argv[])
{
    int sockfd, portno, n;
    struct sockaddr_in serv_addr;
    struct hostent *server;

    if (argc < 3) {
       fprintf(stderr,"usage %s hostname port\n", argv[0]);
       exit(0);
    }
    portno = atoi(argv[2]);
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) 
        error("ERROR opening socket");
    server = gethostbyname(argv[1]);
    if (server == NULL) {
        fprintf(stderr,"ERROR, no such host\n");
        exit(0);
    }
    bzero((char *) &serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    bcopy((char *)server->h_addr, 
         (char *)&serv_addr.sin_addr.s_addr,
         server->h_length);
    serv_addr.sin_port = htons(portno);
    if (connect(sockfd,(struct sockaddr *) &serv_addr,sizeof(serv_addr)) < 0)
        error("ERROR connecting");
    printf("Please enter the message: \n");
    float floatsOut[4];
    cin >> floatsOut[ 0 ];
    for ( unsigned i = 1; i < 4; i++ )
    {
        floatsOut[ i ] = i + 0.04f;
        cout << "floatsOut[" << i <<"]: " << floatsOut[ i ] << endl;
    }
    for ( unsigned i = 0 ;i < 4; i++ )
        printf( "entered [%i]: %f\n ", i, floatsOut[i] );
    unsigned chBuffSize =  4 * si3zeof( float );
    unsigned char chFloats[ chBuffSize ];
    for ( unsigned i = 0; i < 4; i++ )
        memcpy( &chFloats[ i * sizeof( float ) ], ( unsigned char* )( &floatsOut[ i ] ), sizeof( float ) );
    cout << "bufferOut: [";
    for ( unsigned i = 0; i < chBuffSize; i++ )
         cout << chFloats[ i ];
    cout << "]" << endl;
    n = write(sockfd,chFloats,chBuffSize);
//     n = write(sockfd,floatsOut,4);

//     char tmpbuff[4];
//     int ret = snprintf( tmpbuff, sizeof( tmpbuff ), "%f", floatsOut[ 0 ] );
//     n = write(sockfd,tmpbuff,4);
    if (n < 0) 
         error("ERROR writing to socket");
    char bufferOut[ 255 ];
    bzero(bufferOut,255);
    n = read(sockfd,bufferOut,255);
    if (n < 0) 
         error("ERROR reading from socket");
    printf("%s\n",bufferOut);
    close(sockfd);
    return 0;
}
