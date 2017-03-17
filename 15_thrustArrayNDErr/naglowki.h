//STL
#include <iostream>
#include <vector>
#include <time.h>
#include <thread>
//CUDA
#include "../book.h"
//stale kompilatora
#define BLOCK_SIZE 10

using namespace std;

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

// Forward declaration of the matrix multiplication kernel
// __global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
    Cvalue += A.elements[row * A.width + e]
    * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
}
