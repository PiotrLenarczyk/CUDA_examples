//STL
#include <iostream>
#include <sys/shm.h>
#include <sys/stat.h>

key_t key = 01234;
const unsigned array1Size = 4;
const unsigned array2Size = 15;

struct Arrays //only float array
{
	unsigned shmid; 
    float array1 [ array1Size ];
    float array2 [ array2Size ];
};
