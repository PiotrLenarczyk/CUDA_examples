#include <cstdlib>
#include <cstdio>

const int DIMENSION = 10;

extern "C" void exampleHost( float * h, int blockDim, int threadDim );

int main( void )
{
    float * h = ( float * )malloc( DIMENSION * DIMENSION * sizeof( float ) );
    exampleHost( h, DIMENSION, DIMENSION );
    for( int i = 0; i < DIMENSION; i++ )
    {
        for( int j = 0; j < DIMENSION; j++ )
        {
            printf( "%2.0f ", h[ i * DIMENSION + j ] );
        } 
    printf( "\n" );
    }

    return 0;
}
