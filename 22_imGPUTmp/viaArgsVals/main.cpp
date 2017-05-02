//STL
#include <iostream>
#include <vector>
#include <stdlib.h>     //system
#include <string>
#include <sstream>		//ostringstream
#include <iomanip>		//setprecision
/*
http://stackoverflow.com/questions/16605967/set-precision-of-stdto-string-when-converting-floating-point-values
*/
using namespace std;
/*
touch a.out && rm a.out && qmake && make && clear && gdb -ex run ./a.out
*/

unsigned i;

template < typename T >
string to_string_with_precision( const T a_value, const int n = 12 )
{
    ostringstream out;
    out << setprecision( n ) << a_value;
    return out.str();
}

int main( void )
{
    vector< float > vec( 10, 401.1f );
    vec[ 3 ] = 7.1f;
	for ( i = 0; i < ( unsigned )vec.size(); i++ )
    	cout << "vec[" << i << "]: " << vec[ i ] << endl;
    cout << " - - - - - - - - - - - - - - - - - - - - - - - - - - " << endl; 
    cout << "			GPU			" << endl;
    cout << " - - - - - - - - - - - - - - - - - - - - - - - - - - " << endl; 
    string comm( "childProc/a.out " );
    string parentArg;
    for ( i = 0; i < ( unsigned )vec.size(); i++ )
	    parentArg.append( to_string_with_precision( vec[ i ] ) + "_" );
    int comRet=system( comm.append( parentArg ).c_str() );if(comRet!=0){cerr<<"Command error!\n";return -1;}
    cout << " - - - - - - - - - - - - - - - - - - - - - - - - - - " << endl; 
    
    return printf( "Some program output int[%i]\n", (int)5 );
}

