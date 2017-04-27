//STL
#include <iostream>
#include <string>

using namespace std;

int main( int argc, char* argv[] )
{
    string childInput ( argv[ argc - 1 ] );
    cout << "raw input: " << childInput << endl;
    long long unsigned *vecPtr = (long long unsigned*)( stoull( childInput ) );
    cout << "float *: " << vecPtr << endl;
    cout << "... some way to get value *vecPtr ..." << endl;
    cout << "vec[3]: " << *reinterpret_cast< float* >( vecPtr ) << endl;       //core segfault

    return 0;
}

