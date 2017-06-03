//STL
#include <iostream>
#include <vector>
#include <time.h>
#include <algorithm>
//THRUST
#include <thrust/reduce.h>              
// #include <thrust/sequence.h>           
#include <thrust/device_vector.h>       

//h_CPU namespace
using namespace std;
//h_CPU global constants
const unsigned arrSize = 01024000000; 
vector< float > h_arr ( arrSize );
unsigned i;
clock_t t;
//h_CPU function
float floatArrSum( vector< float > &arrIn );
double doubleArrSum( vector< float > &arrIn );

//d_GPU namespace
using namespace thrust;
//d_GPU variables
device_vector< float > d_arr( arrSize );

//MAIN FUNCTION BODY
int main()
{
    for ( i = 0; i < arrSize; i++ )
        h_arr[ i ] = 1.0f;   
    for( unsigned test = 0; test < 1; test++ )
    {
        t = clock();
        float floatCPUResult = floatArrSum( h_arr );
        cout << "floatCPUResult: " << floatCPUResult << endl;
        cout << "float CPU clocks: " << float(( clock() - t ) ) << endl;
        
        double doubleCPUResult = doubleArrSum( h_arr );
        cout << "doubleCPUResult: " << doubleCPUResult << " [double datatype of accumulator]" << endl;
        cout << "double CPU clocks: " << float(( clock() - t ) ) << endl;
        
        d_arr = h_arr;
        t = clock();
        float gpuResult = reduce( d_arr.begin(), d_arr.end() );
        cout << "gpuResult: " << gpuResult << endl;
        cout << "elapsed CPU clocks of GPU computations: " << float(( clock() - t ) ) << endl;
        
        //computations check
        if ( abs( doubleCPUResult - gpuResult ) > 10E-7 )
        {
            cerr << "abs( doubleCPUResult - gpuResult ): " << abs( doubleCPUResult - gpuResult ) << endl;
            cerr << "COMPUTATIONS ERROR" << endl;
            return -1;
        }
        cout << " - - - - - - - - - - " << endl;
    }
    
    return 0;
}

//h_CPU functions
float floatArrSum( vector< float > &arrIn )
{
    float sum( 0.0f );
    for ( i = 0; i < arrSize; i++ )
        sum += arrIn[ i ];
    return sum;
}

double doubleArrSum( vector< float > &arrIn )
{
    double sum( 0.0 );
    for ( i = 0; i < arrSize; i++ )
        sum += arrIn[ i ];
    return sum;
}

//P.S. note gpu high - flexibility in providing code and high-throuhput architecture.
//P.P.S. note horribly gpu-code developing time efficiency for beginners ( during my small experience in gpu programming )
//P.P.P.S. quite suprising results for ultra-sequential basic CPU operations as basic as vector sum. There are no GPU and CPU code optimalizations in this example.
