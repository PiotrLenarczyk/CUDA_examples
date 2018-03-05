//THRUST
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
//STL
#include <algorithm>
#include <cstdlib>
#include <time.h>

using std::cout; using std::endl;

__global__ void emptyKernel( void ){};

int main(void)
{
    // generate random data serially
    thrust::host_vector<int> h_vec(1000000);
    std::generate(h_vec.begin(), h_vec.end(), rand);
    
    // transfer to device and compute sum
    thrust::device_vector<int> d_vec = h_vec;
    clock_t t = clock();
    int x = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<int>());
    emptyKernel<<< 1, 1 >>>();
    cout << "CPU clocks: " << float( clock() - t ) << endl;
    
    return 0;
}
