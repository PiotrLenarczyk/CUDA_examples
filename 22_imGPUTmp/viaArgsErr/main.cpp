//STL
#include <iostream>
#include <vector>
#include <stdlib.h>     //system
#include <string>

using namespace std;
/*

touch a.out && rm a.out && qmake && make && clear && gdb -ex run ./a.out

*/

int main( int argc, char* argv[] )
{
    vector< float > vec( 10, 401.1f );
    cout << "vec[3]: " << vec[ 3 ] << endl;
    cout << "&vec: " << &vec << endl;
    cout << "&vec[0]: " << &vec[ 0 ] << endl;
    float *vecElemPtr = &vec[ 3 ];
    cout << "vecElemPtr: " << vecElemPtr << endl;
    cout << "*vecElemPtr: " << *vecElemPtr << endl;
    cout << "///////////////////////////////////////////////" << endl;
    cout << "reinterpret_cast<long long unsigned >(vecElemPtr): " << to_string( reinterpret_cast<long long unsigned >( vecElemPtr ) )  << endl; 
    string comm( "childProc/childProc " );
    string parentArg( to_string( reinterpret_cast<long long unsigned >(vecElemPtr) ) );
    comm.append( parentArg );
    system( comm.c_str() );
    cout << "///////////////////////////////////////////////" << endl;
    cout << "*(float *)( reinterpret_cast<long long unsigned >( vecElemPtr ) ): " << *(float *)( reinterpret_cast<long long unsigned >( vecElemPtr ) ) << endl;
    cout << " - - - - - - - - - - - - - - - - - - - - - - - - - - " << endl;
    cout << "long long unsigned: " << stoull( parentArg ) << endl;
    cout << "float : " << ( *(float *)(stoull( parentArg )) ) << endl;
    cout << "*reinterpret_cast< float* >( stoull( parentArg ) ): " << *reinterpret_cast< float* >( stoull( parentArg ) ) << endl;       //core segfault
    cout << " - - - - - - - - - - - - - - - - - - - - - - - - - - " << endl;
    
    return printf( "Some program output int[%i]\n", (int)5 );
}

