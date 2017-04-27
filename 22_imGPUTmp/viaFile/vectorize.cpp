//STL
#include <iostream>
//Qt
#include <QVector>
#include <QFile>
#include <QTextStream>
#include <QImage>
#include <QString>

using namespace std;

//global variables
QVector < QString > imName;
QVector < QImage > imVec;
QImage imSingleton;
int i, x, y, X, Y;

int main( void )
{
    imName.push_back( "lena.jpg" );
    if ( !imSingleton.load( imName[ 0 ] ) ) { cerr << "Image opening error!\n"; return -1; }
    imVec.push_back( imSingleton );
    QFile imagesFile( "gpuCode/imFile.txt" );
    for ( i = 0; i < imVec.size(); i++ )
    {
        X = imVec[ i ].width();
        Y = imVec[ i ].height();
        QRgb *lineTmp;
        if ( imagesFile.open( QFile::WriteOnly | QFile::Truncate ) )
        {
            QTextStream out ( &imagesFile );
            out.setRealNumberPrecision( 12 );
            out << X << " " << Y << endl;
            for ( y = 0; y < Y; y++ )
            {
                lineTmp = (QRgb *)imVec[ i ].scanLine( y );
                for ( x = 0; x < X; x++ )
                      out << ( 16.0f + ( 1.0f / 256.0f ) * ( 
                                    65.738f * qRed( lineTmp[ x ] ) + 
                                    129.057f * qGreen( lineTmp[ x ] ) +
                                    25.064f * qBlue( lineTmp[ x ] ) ) )
                      << " ";
                     
            }
            out << endl;
        }
    }
    
    return 0;
}

//P.S. please note some tricks for files IO's like for example RAMDISKS.
