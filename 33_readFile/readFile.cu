#include <fstream>
#include <stdio.h>
#include <iostream>
#define H2D cudaMemcpyHostToDevice 
#define D2H cudaMemcpyDeviceToHost
#define OK cudaSuccess

using namespace std;
typedef uint32_t uint;
typedef unsigned char BYTE;

//CPU
uint i = 0, ind = 0;
const uint N = 8E3;
//GPU

__global__ void emptyKernel( BYTE *d_in )
{	printf( "gpu read file:[" );
	for ( uint i = 0; i < 4; i++ )
		printf( "%X", d_in[ i ] );
	printf( "...]\n" );
};


int main( void )
{
//	write to file
	BYTE outputToFile[ N ]; memset( &outputToFile, 0xAB, N );
	uint fileSize = N * sizeof( BYTE );
	char fileName[] = "tmpBinaryFile.txt";
	ofstream outFile( fileName, ofstream::binary );  //|ofstream::app
		outFile.write( ( char* )&outputToFile[ 0 ], fileSize );
	outFile.close();
	printf( "fileSize : %i\n", fileSize );
	printf( "saved HDD file:[" );
	for ( i = 0; i < 4; i++ )
		printf( "%X", outputToFile[ i ] );
	printf( "...]\n" );
	
//	read from file
	ifstream inFile( fileName, ifstream::binary );
		inFile.seekg( 0, inFile.end ); uint fileSizeRead = inFile.tellg(); 
		printf( "fileSizeRead : %i\n", fileSizeRead );
		inFile.seekg( 0, inFile.beg );
		BYTE *inputFromFile = ( BYTE* )malloc( fileSizeRead );
		inFile.read( ( char* )inputFromFile, fileSizeRead );
	inFile.close();
	printf( "read HDD file:[" );
	for ( i = 0; i < 4; i++ )
		printf( "%X", inputFromFile[ i ] );
	printf( "...]\n" );
//		copy file to GPU
	BYTE* d_fromFile[ 1 ];
	if ( cudaMalloc( ( void** )&d_fromFile[ 0 ], fileSize ) != cudaSuccess ) { printf( "cudaMalloc err!\n" ); return -1; };
	cudaMemcpyAsync( d_fromFile[ 0 ], inputFromFile, fileSizeRead, H2D );
	emptyKernel<<< 1, 1 >>>( d_fromFile[ 0 ] );
	
	delete( inputFromFile );
	cudaFree( d_fromFile[ 0 ] );

	return 0;
}; //end of main()
