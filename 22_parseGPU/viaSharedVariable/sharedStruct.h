typedef struct sharedData* dataPtr;
const unsigned pics = 4;
struct sharedData
{
    long structShmid;
    unsigned int size;
    unsigned int picsX[ pics ];
    unsigned int picsY[ pics ];
    float value[ 8388608 * pics ]; //default allocation for 4096x2048
};
struct sharedData* data;
