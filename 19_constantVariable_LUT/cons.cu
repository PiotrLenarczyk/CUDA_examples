//  http://cuda-programming.blogspot.com/2013/01/what-is-constant-memory-in-cuda.html
 //declare constant memory
__constant__ float cangle[360]; //LUT candidate

__global__ void test_kernel(float* darray);

int main(int argc,char** argv)
{
         int size=3200;
         float* darray;
         float hangle[360];

          //allocate device memory
         cudaMalloc ((void**)&darray,sizeof(float)*size);
         
         //initialize allocated memory
        cudaMemset (darray,0,sizeof(float)*size);

         //initialize angle array on host
       for(int loop=0;loop<360;loop++)
                    hangle[loop] = acos( -1.0f )* loop/ 180.0f;

        //copy host angle data to constant memory
       cudaMemcpyToSymbol    (  cangle,  hangle,   sizeof(float)*360  );
   
        test_kernel  <<<  size/64  ,64  >>>  (darray);
     
       //free device memory
       cudaFree(darray);
  return 0;
}


__global__ void test_kernel(float* darray)
{
      int index;
    
      //calculate each thread global index
     index = blockIdx.x * blockDim.x + threadIdx.x;
   
    #pragma unroll 10
   for(int loop=0;loop<360;loop++)
                 darray[index]= darray [index] + cangle [loop] ;
return;

}
