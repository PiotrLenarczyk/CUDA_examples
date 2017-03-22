// http://stackoverflow.com/questions/13669019/performing-fourier-transform-with-thrust
//STL
#include <iostream>
#include <stdlib.h>
//THRUST
#include <thrust/device_vector.h>
#include <thrust/for_each.h>            //normalize with size inversed FFT
//cuFFT
#include <cufft.h>

using namespace thrust;

__constant__ unsigned N_dev[ 1 ]; //constant device variable declaration

struct normalize_iFFT_functor
{
  __host__ __device__ void operator()(float &x)
  { 
      x /= N_dev[ 0 ];
  }
};

int main(void){
    unsigned N = 7; cudaMemcpyToSymbol( N_dev[ 0 ], &N, sizeof( unsigned ) ); //constant device variable HtD
    unsigned Nfft = ( N / 2 ) + 1; //non - redundant FFT
    
//  //FFT Simple data ( N ) Real-To-Complex spectrum ( N/2 + 1 )
    // --- Setting up input device vector    
    device_vector< float > d_in( N, 1.f ); 
    device_vector< cuFloatComplex > d_out( Nfft );
    for ( unsigned i = 0; i < N; i++ ) 
        printf( "Orig [ #%i ]: Real part = %f\n", i, float( d_in[ i ] ) );
    
    cufftHandle planFFTSimple;    
    cufftPlan1d( &planFFTSimple, N, CUFFT_R2C, 1 );
    cufftExecR2C( planFFTSimple, raw_pointer_cast( d_in.data() ), raw_pointer_cast( d_out.data() ) );
    for ( unsigned i = 0; i < Nfft; i++ ) 
        printf( "FFT_1D [ #%i ]: Real part = %f; Imaginary part: %f\n", i, cuFloatComplex( d_out[ i ] ).x, cuFloatComplex( d_out[ i ] ).y );
    
//  //iFFT inverse spectrum ( N/2 + 1 ) Complex-To-Real data( N ) inversed
    device_vector< float > d_inversed( N );
    cufftHandle planFFTInverse;    
    cufftPlan1d( &planFFTInverse, N, CUFFT_C2R, 1 );
    cufftExecC2R( planFFTInverse, raw_pointer_cast( d_out.data() ), raw_pointer_cast( d_inversed.data() ) );
    for_each( d_inversed.begin(), d_inversed.end(), normalize_iFFT_functor() ); //normalize with size iFFT results
    for ( unsigned i = 0; i < N; i++ ) 
        printf( "IFFT_1D [ #%i ]: Real part = %f\n", i, float ( d_inversed[ i ] ) );

    cudaFree( N_dev );  //constant device variable memory deallocation
    cudaDeviceSynchronize();
    return 0;
}
