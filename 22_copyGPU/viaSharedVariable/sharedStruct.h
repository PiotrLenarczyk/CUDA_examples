typedef struct sharedData* dataPtr;
const unsigned picAllocX = 3584;
const unsigned picAllocY = 2016;
const unsigned pics = 4;
struct sharedData
{
    long structShmid;
    unsigned picsNo;
    unsigned int size;
    unsigned int picsX[ pics ];
    unsigned int picsY[ pics ];
    float value[ picAllocX * picAllocY * pics ]; //default allocation for 4096x2048 up to 7MPix images
};
struct sharedData* data;
