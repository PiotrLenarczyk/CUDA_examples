#include "device.h"

/*
 
/usr/local/cuda-8.0/bin/nvcc -Wno-deprecated-gpu-targets -O2 -c device.cu && g++  -O2 -c host.cpp -I/usr/local/cuda/include/ && /usr/local/cuda-8.0/bin/nvcc -Wno-deprecated-gpu-targets -o tester device.o host.o && ./tester dataFile.txt 

*/
using namespace std;

#include <string>
#include <fstream>
struct Reading
{
    float hour;
    float temperature;
    Reading( float h, float t ) :hour( h ), temperature( t ) { }
};

int main( int argc, char *argv[] )
{
    //read from file
    string name = argv[ argc - 1 ];
    ifstream ist( name.c_str() );
    if ( !ist )
    {
        cerr << "Input file open error: " << name << " !" << endl;
        return -1;
    }
    vector< Reading > temps;
    float hour;
    float temperature;
    while ( ( ist >> hour >> temperature ) && ( ist.good() ) )
    {
        if ( hour < 0 || 23 < hour )
            cerr << "Hour value err!";
        temps.push_back( Reading( hour, temperature ) );
    }
 
    int h_vecSize = ( int )temps.size();
    thrust::host_vector< float > h_vecFile( h_vecSize, 0.0f );
    int h_vecDims = 3;
    thrust::host_vector< thrust::host_vector< float > > h_vecFile2D( h_vecDims, h_vecFile );
    for ( int i = 0; i < ( int )temps.size(); i++ )
    {
        cout << "( " << temps[ i ].hour << ", " << temps[ i ].temperature << " )" << endl;
        h_vecFile2D[ 0 ][ i ] = temps[ i ].hour ;
        h_vecFile2D[ 1 ][ i ] = temps[ i ].temperature;
    }    
    cout << "Host vector hour: \n";
    thrust::copy( h_vecFile2D[0].begin(), h_vecFile2D[0].end(), std::ostream_iterator< float >( std::cout, "\n" ) );
    cout << "Host vector temperature: \n";
    thrust::copy( h_vecFile2D[1].begin(), h_vecFile2D[1].end(), std::ostream_iterator< float >( std::cout, "\n" ) );
    
    vec1DTH ( h_vecFile ); //1D vector copy to GPU memory
    
    thrust::host_vector< float > h_vecSorted( h_vecSize, 0.0f );
    for ( int i = 0; i < h_vecDims; i++ )
    {
        h_vecSorted = h_vecFile2D[ i ];
        sort_on_device( h_vecSorted );
        if ( i == 1 )
        {
            cout << "Sorted host vector temperature: \n";
            thrust::copy( h_vecSorted.begin(), h_vecSorted.end(), std::ostream_iterator< float >( std::cout, "\n" ) );
        }
    }


    return 0;
}
