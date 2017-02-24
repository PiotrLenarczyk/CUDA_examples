__global__ void exampleDevice( float * d )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    d[ idx ] = idx;   
}

extern "C" void exampleHost( float * h, int blockDim, int threadDim )
{
    float * d;
    cudaMalloc( ( void** )&d, blockDim * threadDim * sizeof( float ) );
    exampleDevice<<<blockDim, threadDim>>>( d );
    cudaMemcpy( h, d, blockDim * threadDim * sizeof( float ), cudaMemcpyDeviceToHost ); 
}
