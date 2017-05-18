# Build tools 
FLAGS=-Wno-deprecated-gpu-targets -std=c++11 -O3
DynParFLAGS = -arch=sm_35 -rdc=true 
NVLIBS = -L /usr/lib/x86_64-linux-gnu/ -lcudadevrt
#NVCC = `find /usr -name 'nvcc' -print -quit` $(FLAGS) $(DynParFLAGS) $(NVLIBS)
NVCC = /usr/local/cuda-8.0/bin/nvcc $(FLAGS) $(DynParFLAGS) $(NVLIBS)

# here are all the objects
GPUOBJS = fp3.o 

# make and compile
a.out:  $(GPUOBJS)
	$(NVCC) -o a.out $(GPUOBJS) 

fp3.o: fp3.cu
	$(NVCC) -c fp3.cu

clean:
	rm fp3.o

delete:
	rm a.out
