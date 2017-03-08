//STL
#include <iostream>
#include <math.h>
#include <vector>
#include <thread>
#include <time.h>
//CUDA
#include <cuda_runtime.h>
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

//przestrzenie nazw
using namespace std;

//PROGRAM GLOBALS
const int N = 8000;

//HOST + GPU GLOBALSY
extern vector < int > acc;
extern vector < vector < int > > acc2;
extern vector < vector < float > > firstMat;
extern vector < vector < float > > secondVecMat;
extern vector < vector < float > > resultsHostMat;
extern vector < vector< float > > resultsGPU;
extern vector < float > firstVec;
extern vector < float > secondVec;
extern vector < float > resultsHost;
extern vector < float > resGPU;
extern float tmp;

//CUDA FUNCTIONS
extern "C" cudaError_t cuda_main( int &iter );
extern "C" void memInit();
extern "C" void memFree();
extern "C" void dodajMatJadro( void *iter );
