//STL
#include <stdio.h>

__global__ void childKernel()
{
    printf( "Hello %d", threadIdx.x );
}

__global__ void parentKernel()
{
    childKernel<<< 1, 2 >>>();
    cudaDeviceSynchronize();
    printf( "World!\n" );
}

int main( int argc, char *argv[] )
{
    parentKernel<<< 1, 2 >>>();
    cudaDeviceSynchronize();
    return 0;
}
