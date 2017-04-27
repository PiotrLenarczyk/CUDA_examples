//STL
#include <iostream>
#include <fstream>

//Qt
#include <QVector>
#include <QImage>
#include <QString>

using namespace std;

//global variables
QVector < QString > imName;
QVector < QImage > imVec;
QImage imSingleton;
unsigned imNo, i, x, y, X, Y;

int main( void )
{
    imName.push_back( "lena.jpg" );
    if ( !imSingleton.load( imName[ 0 ] ) ) { cerr << "Image opening error!\n"; return -1; }
    imVec.push_back( imSingleton );
    imNo = ( unsigned )imVec.size();
    
    ofstream imagesFile;    //2*intSizeYX[image]; YX*floatLuminance[image];
    imagesFile.open( "gpuCode/imFile.txt", ios::out | ios::binary );
    imagesFile.write( reinterpret_cast< char* >( &imNo ), 1 * sizeof( unsigned ) ); 
    for ( i = 0; i < (unsigned)imVec.size(); i++ )
    {
        X = imVec[ i ].width();
        Y = imVec[ i ].height();
        vector < float > imY( X * Y, 0.0f );
        QRgb *lineTmp;
        imagesFile.write( reinterpret_cast< char* >( &Y ), 1 * sizeof( unsigned ) ); 
        imagesFile.write( reinterpret_cast< char* >( &X ), 1 * sizeof( unsigned ) ); 
        for ( y = 0; y < Y; y++ )
        {
            lineTmp = (QRgb *)imVec[ i ].scanLine( y );
            for ( x = 0; x < X; x++ )
            {
                imY[ y * Y + x ] = ( 16.0f + ( 1.0f / 256.0f ) * ( 
                                65.738f * qRed( lineTmp[ x ] ) + 
                                129.057f * qGreen( lineTmp[ x ] ) +
                                25.064f * qBlue( lineTmp[ x ] ) ) );
                
            }
        }
        imagesFile.write( reinterpret_cast< char* >( &imY[ 0 ] ), imY.size() * sizeof( float ) ); 
    }
    imagesFile.close();
    
    return 0;
}

//P.S. please note some tricks for files IO's like for example RAMDISKS.
