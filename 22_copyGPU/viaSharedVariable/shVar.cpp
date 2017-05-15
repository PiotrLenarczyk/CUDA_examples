/* http://advancedlinuxprogramming.com/ */
/* https://ubuntuforums.org/archive/index.php/t-1426536.html */
//STL
#include <QImage>
#include <iostream>
#include <sys/shm.h>
#include <sys/stat.h>
#include <stdlib.h>
#include "sharedStruct.h"
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
float tmp;
//functions
void closeSHM( long destroySHM );
void freeSHM( struct sharedData* data );
void loadPicsLuminances();

int main( void )
{   
    key_t key = 1234;
    int shmid = shmget( key, sizeof( struct sharedData ), IPC_CREAT | 0666 );

    if ( shmid < 0 )
    {
        cerr << "shmget ERROR!\n";
        return -1;
    }
    data = ( struct sharedData* )  shmat( shmid, NULL, 0 );
    data->structShmid = shmid;
    cout << "data size: " << sizeof( struct sharedData ) / ( 1024.0 * 1024.0 ) << "[MB]" << endl;
    if ( ( long )data == -1 )
    {
        cerr << "shmat ERROR!\n";
        return -1;
    }
    
    loadPicsLuminances();
    
    if ( system( "./aChild.out" ) != 0 ) { cerr << "GPU execution error!\n"; return -1; }
    
//     long destroyShm = data->structShmid;
    freeSHM( data ); //closeSHM( destroyShm );
    return 0;
}

void freeSHM( struct sharedData* data )
{
    if ( shmdt( data ) < 0 )
        cerr << "child shmdt ERROR!\n";
}

void closeSHM( long destroySHM )
{
    if ( shmctl( destroySHM, IPC_RMID, NULL ) < 0 )
        cerr << "child shmctl ERROR!\n";
}

void loadPicsLuminances()
{
    
    imName.push_back( "lena.jpg" ); imName.push_back( "baboon.png" ); imName.push_back( "barbara.jpg" );
    for ( i = 0; i < ( unsigned )imName.size(); i++ ) 
        if ( !imSingleton.load( imName[ i ] ) ) cerr << "Image [" << imName[ i ].toStdString() << "]opening error!\n";
        else
            imVec.push_back( imSingleton );
    imNo = ( unsigned )imVec.size();    
    unsigned overallSize = 0; unsigned indData = 0;
    for ( i = 0; i < (unsigned)imVec.size(); i++ )
    {
        X = imVec[ i ].width();
        Y = imVec[ i ].height();
        overallSize += X * Y;
        vector < float > imY( X * Y, 0.0f );
        QRgb *lineTmp;
        data->picsX[ i ] = X;
        data->picsY[ i ] = Y;
        for ( y = 0; y < Y; y++ )
        {
            lineTmp = (QRgb *)imVec[ i ].scanLine( y );
            for ( x = 0; x < X; x++ )
            {
                tmp = ( 16.0f + ( 1.0f / 256.0f ) * ( 
                                65.738f  * qRed( lineTmp[ x ] ) + 
                                129.057f * qGreen( lineTmp[ x ] ) +
                                25.064f  * qBlue( lineTmp[ x ] ) ) );
                imY[ y * Y + x ] = tmp;
                data->value[ indData ] = tmp;
                indData++;
            }
        }
        for ( unsigned ii = 0; ii < 3; ii++ )
            cout << "imY[ " << i << " ][ " << ii << " ]: " << imY[ ii ] << endl;
    }
    data->size = overallSize;
}
