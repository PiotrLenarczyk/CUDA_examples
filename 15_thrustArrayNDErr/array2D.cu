#include <cstdlib>
#include <iostream>

const unsigned int N( 16 );

using std::cout; using std::endl;

__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]) //dot mat addition
{
int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;
if (i < N && j < N)
    C[i][j] = A[i][j] + B[i][j];
printf( "[%02i][%02i]: %.2f = %.2f + %.2f\n", i, j, C[i][j], A[i][j], B[i][j] );
}

int main( void )
{
    float A[N][N] = { 0.0f };
    float B[N][N] = { 0.1f };
    float C[N][N] = { 0.0f };
    cout << "A[0][0]: " << A[0][0] << endl;
    cout << "B[0][0]: " << B[0][0] << endl;    
    cout << "C[0][0]: " << C[0][0] << endl;

    float (*d_A)[N]; //pointers to arrays of dimension N
    float (*d_B)[N];
    float (*d_C)[N];
    

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            A[i][j] = i;
            B[i][j] = j;
        }
    }       

    //allocation
    cudaMalloc((void**)&d_A, (N*N)*sizeof(float));
    cudaMalloc((void**)&d_B, (N*N)*sizeof(float));
    cudaMalloc((void**)&d_C, (N*N)*sizeof(float));

    //copying from host to device
    cudaMemcpy(d_A, A, (N*N)*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, (N*N)*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, (N*N)*sizeof(float), cudaMemcpyHostToDevice);

    // Kernel invocation
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    MatAdd<<< numBlocks, threadsPerBlock >>>(d_A, d_B, d_C);

    //copying from device to host
//     cudaMemcpy(A, (d_A), (N*N)*sizeof(float), cudaMemcpyDeviceToHost);
//     cudaMemcpy(B, (d_B), (N*N)*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(*C, (d_C), (N*N)*sizeof(float), cudaMemcpyDeviceToHost);
    
    cout << "A[0][0]: " << A[0][0] << endl;
    cout << "B[0][0]: " << B[0][0] << endl;  
    cout << "C[0][0]: " << C[0][0] << endl;

    cudaDeviceSynchronize();
    return 0;
}
