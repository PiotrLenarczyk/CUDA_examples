# Build tools 
FLAGS=-Wno-deprecated-gpu-targets -std=c++11
DynParFLAGS = -arch=sm_35 -rdc=true 
NVLIBS = -L /usr/lib/x86_64-linux-gnu/ -lcudadevrt
#NVCC = `find /usr -name 'nvcc' -print -quit` $(FLAGS) $(DynParFLAGS) $(NVLIBS)
NVCC = /usr/local/cuda-8.0/bin/nvcc $(FLAGS) $(DynParFLAGS) $(NVLIBS)

# here are all the objects
GPUOBJS = cons.o 

# make and compile
a.out:  $(GPUOBJS)
	$(NVCC) -o a.out $(GPUOBJS) 

cons.o: cons.cu
	$(NVCC) -c cons.cu

clean:
	rm cons.o

delete:
	rm a.out
