#include "naglowki.h"

//deklaracje funkcji
void MatMul( Matrix &A, Matrix &B, Matrix &C);

int main()
{    
    clock_t t, t1;
    int N = 1000;
    vector < vector < float > > pierwszyMat( N );
    vector < vector < float > > drugiVecMat( N );
    vector < vector < float > > wynikHostMat( N );
    vector < vector< float > > wynikGPUMat( N );
    vector < float > pierwszyVec( N, 3.14 );
    vector < float > drugiVec( N, 2.72 );
    vector < float > wynikHost( N );
    vector < float > wynikGPU( N, 0 ); 
    for ( int i = 0; i < N; i++ )
    {
        pierwszyMat[ i ] = pierwszyVec;
        drugiVecMat[ i ] = drugiVec;
        wynikGPUMat[ i ] = wynikGPU;
        wynikHostMat[ i ] = wynikHost;
    }
    
    t = clock();
    for ( int j = 0; j < N; j++ )
        for ( int i = 0; i < N; i++ )
            wynikGPUMat[ j ][ i ] = pierwszyMat[ j ][ i ] + drugiVecMat[ j ][ i ];    
    t1 = clock() - t;
    cout << "czas obl. sekw. Host: " << ((float)(t1))/CLOCKS_PER_SEC << "[s] ( NVCC )" << endl;

    float AMat[ N * N ];
    for ( int i = 0; i < N * N; i++ )
        AMat[ i ] = 3.14;
    Matrix A;
    A.width = N;
    A.height = N;
    A.elements = &AMat[ 0 ];

    float BMat[ N * N ];
    for ( int i = 0; i < N * N; i++ )
        BMat[ i ] = 2.72;
    Matrix B;
    B.width = N;
    B.height = N;
    B.elements = &BMat[ 0 ];

    vector<float> CMat( N * N );
    for ( int i = 0; i < N * N; i++ )
        CMat[ i ] = 0;
    Matrix C;
    C.width = N;
    C.height = N;
    C.elements = &CMat[ 0 ];
    
    t = clock();
    MatMul( A, B, C );
    t1 = clock() - t;
    cout << "czas obl. sekw. GPU: " << ((float)(t1))/CLOCKS_PER_SEC << "[s] ( NVCC )" << endl;
    
    cudaDeviceReset();
    return 0;
}

// Thread block size
// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul( Matrix &A, Matrix &B, Matrix &C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
    cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
    cudaMemcpyHostToDevice);
    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);
    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
    cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}
