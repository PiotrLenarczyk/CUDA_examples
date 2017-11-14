# Build tools 
FLAGS=-Wno-deprecated-gpu-targets -std=c++11
DynParFLAGS = -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_61,code=sm_61 -rdc=true  
NVLIBS = -lcudadevrt 
NVINC = -I /usr/local/cuda-8.0/include/thrust -I /usr/local/cuda-8.0/targets/x86_64-linux/include/thrust
NVCC = /usr/local/cuda-8.0/bin/nvcc $(FLAGS) $(DynParFLAGS) $(NVLIBS) $(NVINC)

# here are all the objects
GPUOBJS = childProcessing.o 

# make and compile
a.out:  $(GPUOBJS)
	$(NVCC) -o a.out $(GPUOBJS) 

childProcessing.o: childProcessing.cu
	$(NVCC) -c childProcessing.cu

clean:
	rm childProcessing.o

delete:
	touch a.out && rm a.out
