#include <iostream>
#include <stdio.h>

using namespace std;

int main ( void )
{
	cudaDeviceProp prop;
	int count;
	cudaGetDeviceCount( &count );
	for ( int i = 0; i < count; i++ )
	{
		cudaGetDeviceProperties( &prop, i );
		printf( "=======================================================================================================\n" );
        printf( "========= 7D GPU of 6D GX [ 4D BX; 5D BY; 6D BZ ] of 4D B [ 1D THX; 2D THY; 3D THZ ] ==================\n" );
		printf( "========= single Thread[] in 4D Block [ THX, THY, THZ ] acts as ALU ===================================\n" );                
		printf( "=======================================================================================================\n" );                
		printf( "Grid[ Blocks, Blocks, Blocks ] => gridKnot/kernel<<< Blocks, BlockThread >>>\n" );
		printf( "Grid[ Blocks, Blocks, Blocks ] => gridKnot/kernel<<< Block, BlockThreads >>>\n" );                
		printf( "Grid[ 0,0,0 ] = Block0[ Thread[ 0, 0, 0 ], Thread[ 1, 0, 0 ], Thread[ 2, 0, 0 ], Thread[ 3, 0, 0 ]... ]\n" );
		printf( "Grid[ 0,0,0 ] = Block0[ Thread[ 0, 0, 0 ], Thread[ 0, 1, 0 ], Thread[ 0, 2, 0 ], Thread[ 0, 3, 0 ]... ]\n" );
		printf( "Grid[ 0,0,0 ] = Block0[ Thread[ 0, 0, 0 ], Thread[ 0, 0, 1 ], Thread[ 0, 0, 2 ], Thread[ 0, 0, 3 ]... ]\n" );
		printf( "Grid[ 1,0,0 ] = Block1[ Thread[ 0, 0, 0 ], Thread[ 1, 0, 0 ], Thread[ 2, 0, 0 ], Thread[ 3, 0, 0 ]... ]\n" );
		printf( "Grid[ 1,0,0 ] = Block1[ Thread[ 0, 0, 0 ], Thread[ 0, 1, 0 ], Thread[ 0, 2, 0 ], Thread[ 0, 3, 0 ]... ]\n" );
		printf( "Grid[ 1,0,0 ] = Block1[ Thread[ 0, 0, 0 ], Thread[ 0, 0, 1 ], Thread[ 0, 0, 2 ], Thread[ 0, 0, 3 ]... ]\n" );
		printf( "Grid[ 2,0,0 ] = Block2[ Thread[ 0, 0, 0 ], Thread[ 1, 0, 0 ], Thread[ 2, 0, 0 ], Thread[ 3, 0, 0 ]... ]\n" );
		printf( "Grid[ 2,0,0 ] = Block2[ Thread[ 0, 0, 0 ], Thread[ 0, 1, 0 ], Thread[ 0, 2, 0 ], Thread[ 0, 3, 0 ]... ]\n" );
		printf( "Grid[ 2,0,0 ] = Block2[ Thread[ 0, 0, 0 ], Thread[ 0, 0, 1 ], Thread[ 0, 0, 2 ], Thread[ 0, 0, 3 ]... ]\n" );
		printf( "Grid[ 0,1,0 ] = Block3[ Thread[ 0, 0, 0 ], Thread[ 1, 0, 0 ], Thread[ 2, 0, 0 ], Thread[ 3, 0, 0 ]... ]\n" );
		printf( "Grid[ 0,1,0 ] = Block3[ Thread[ 0, 0, 0 ], Thread[ 0, 1, 0 ], Thread[ 0, 2, 0 ], Thread[ 0, 3, 0 ]... ]\n" );
		printf( "Grid[ 0,1,0 ] = Block3[ Thread[ 0, 0, 0 ], Thread[ 0, 0, 1 ], Thread[ 0, 0, 2 ], Thread[ 0, 0, 3 ]... ]\n" );
		printf( "===============================================================================\n" );
        printf( "Max. Blocks per Grid: 2^31 - 1\n" );
        printf( "Max. Threads per Block: %d\n", prop.maxThreadsPerBlock );        
		printf( "Max. No. of Thread Dimensions: [ %d, %d, %d ]\n", prop.maxThreadsDim[ 0 ], prop.maxThreadsDim[ 1 ], prop.maxThreadsDim[ 2 ] );
		printf( "Max. No. of Grid Dimensions: [ %d, %d, %d ]; including max. [ %d ] 1D-3D threads per block \n", 
				prop.maxGridSize[ 0 ], prop.maxGridSize[ 1 ], prop.maxGridSize[ 2 ], prop.maxThreadsPerBlock );
		printf( "CUDA ver.: %d.%d\n", prop.major, prop.minor );
        printf( "Integrated GPU.: %d\n", prop.integrated );             
        printf( "Concurrency: %d\n", prop.concurrentKernels );
        printf( "Stream Multiprocesors SMM.: %d\n", prop.multiProcessorCount );
	}
	
	return 0;
}
