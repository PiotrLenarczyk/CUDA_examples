//http://www.orangeowlsolutions.com/archives/1153
#include <stdio.h>

#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

/*

nvcc -Wno-deprecated-gpu-targets -std=c++11 -arch=sm_35 -rdc=true  -L /usr/lib/x86_64-linux-gnu/ -lcudadevrt -lcufft -I /usr/include/thrust/  thrustKernel.cu && ./a.out 

*/

/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %dn", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void test(float *d_A, int N) {

    float sum = thrust::reduce(thrust::seq, d_A, d_A + N);

    printf("Device side result = %f \n", sum);

}

int main() {

    const int N = 16;

    float *h_A = (float*)malloc(N * sizeof(float));
    float sum = 0.f;
    for (int i=0; i<N; i++) {
        h_A[i] = i;
        sum = sum + h_A[i];
    }
    printf("Host side result = %f \n", sum);

    float *d_A; gpuErrchk(cudaMalloc((void**)&d_A, N * sizeof(float)));
    gpuErrchk(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    test<<<1,1>>>(d_A, N);
    cudaDeviceSynchronize();
    return 0;
}
