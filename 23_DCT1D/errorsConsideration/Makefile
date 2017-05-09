# Build tools 						maximize calculations precision
FLAGS=-Wno-deprecated-gpu-targets -std=c++11 --ftz=false --prec-div=true --prec-sqrt=true --fmad=false 
DynParFLAGS = -arch=sm_35 -rdc=true 
NVLIBS = -L /usr/lib/x86_64-linux-gnu/ -lcudadevrt
#NVCC = `find /usr -name 'nvcc' -print -quit` $(FLAGS) $(DynParFLAGS) $(NVLIBS)
NVCC = /usr/local/cuda-8.0/bin/nvcc $(FLAGS) $(DynParFLAGS) $(NVLIBS)

# here are all the objects
GPUOBJS = errorsCons.o 

# make and compile
a.out:  $(GPUOBJS)
	$(NVCC) -o a.out $(GPUOBJS) 

errorsCons.o: errorsCons.cu
	$(NVCC) -c errorsCons.cu

clean:
	rm errorsCons.o

delete:
	rm a.out
