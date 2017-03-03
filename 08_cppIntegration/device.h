#pragma once
//THRUST
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
//STL
#include <cstdlib>
#include <iostream>
#include <iterator>

using namespace thrust;

// function prototype
void sort_on_device( thrust::host_vector< float >& V );
void vec1DTH( thrust::host_vector< float >& h_vec );
