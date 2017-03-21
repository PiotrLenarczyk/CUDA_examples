// http://stackoverflow.com/questions/13669019/performing-fourier-transform-with-thrust
//STL
#include <iostream>
#include <cufft.h>
#include <stdlib.h>
//THRUST
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/transform.h>

using namespace thrust;

int main(void){

    unsigned N = 7;
    unsigned Nfft = N / 2 + 1; //non - redundant

    // --- Setting up input device vector    
    device_vector< cufftReal > d_in( N, 1.f ); 
    device_vector< cuFloatComplex > d_out( Nfft );

    cufftHandle plan;    
    cufftPlan1d( &plan, N, CUFFT_R2C, 1 );
    cufftExecR2C( plan, raw_pointer_cast( d_in.data() ), raw_pointer_cast( d_out.data() ) );

    // --- Setting up output host vector    
    host_vector< cuFloatComplex > h_out( d_out );

    for ( unsigned i = 0; i < Nfft; i++ ) 
        printf( "FFT1D [ #%i ]: Real part = %f; Imaginary part: %f\n", i, h_out[ i ].x, h_out[ i ].y );

    return 0;
}
