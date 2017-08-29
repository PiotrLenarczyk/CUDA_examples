#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>

int main(void)
{
// generate 32M random numbers serially
thrust::host_vector<int> h_vec(32 << 20);
std::generate(h_vec.begin(), h_vec.end(), rand);
thrust::host_vector<int> h_check = h_vec;

// transfer data to the device
thrust::device_vector<int> d_vec = h_vec;

// transfer data back to host
thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());


uint8_t flag = 0;
for ( size_t i = 0; i < h_vec.size(); i++ )
    if ( h_vec[ i ] != h_check[ i ] )
    {
        std::cerr << "Vector check error!\n";
        flag = 1;
        break;
    }
if ( flag == 0 )
    std::cout << "Vector check OK!\n";

// sort data on the device (846M keys per second on GeForce GTX 480)
thrust::sort(d_vec.begin(), d_vec.end());

// transfer data back to host
thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

return 0;
}
