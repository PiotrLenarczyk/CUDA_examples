# Build tools 
FLAGS=-Wno-deprecated-gpu-targets -std=c++11
DynParFLAGS = -arch=sm_35 -rdc=true 
NVLIBS = -lcudadevrt 
NVINC = -I /usr/local/cuda-8.0/include/thrust -I /usr/local/cuda-8.0/targets/x86_64-linux/include/thrust/
NVCC = /usr/local/cuda-8.0/bin/nvcc $(FLAGS) $(DynParFLAGS) $(NVLIBS) $(NVINC)

# here are all the objects
GPUOBJS = imProcess.o 

# make and compile
a.out:  $(GPUOBJS)
	$(NVCC) -o a.out $(GPUOBJS) 

imProcess.o: imProcess.cu
	$(NVCC) -c imProcess.cu

clean:
	rm imProcess.o

delete:
	touch a.out && rm a.out
