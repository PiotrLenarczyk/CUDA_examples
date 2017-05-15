//STL
#include <iostream>
#include <string>
#include <vector>

using namespace std;

string childInput;
unsigned i;
vector < float > inputVec;
string letter, subFp; const string sep( "_" );

//===========================	gpu ===========================
__device__ float d_Array[ 30 ]; 	//static gpu array
__global__ void printKernel()
{
	unsigned ind = threadIdx.x;
	printf( "d_Array[%i]: %f\n", ind, d_Array[ ind ] );
}

int main( int argc, char* argv[] )
{
    childInput = argv[ argc - 1 ];
	for ( i = 0; i < ( unsigned )childInput.size(); i++ )
	{
		letter = childInput[ i ];
    	if ( letter.compare( sep ) != 0 )
    		subFp.append( letter );
    	else
    	{
    		inputVec.push_back( stof( subFp ) );
    		subFp.clear();
    	}
	}
	cudaMemcpyToSymbol( d_Array, &inputVec[ 0 ], sizeof( float ) * ( unsigned )inputVec.size() );
	printKernel<<< 1, (unsigned)inputVec.size() >>> ();
	
	cudaFree( d_Array );
	cudaDeviceSynchronize();
	cudaDeviceReset();
    return 0;
}

