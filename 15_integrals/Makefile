# Build tools 
FLAGS=-Wno-deprecated-gpu-targets -std=c++11 -O3 --fmad=true
DynParFLAGS = -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_61,code=sm_61 -rdc=true 
NVLIBS = -L /usr/lib/x86_64-linux-gnu/ -lcudadevrt
NVCC = `find /usr -name 'nvcc' -print -quit` $(FLAGS) $(DynParFLAGS) $(NVLIBS)
# NVCC = /usr/local/cuda-8.0/bin/nvcc $(FLAGS) $(DynParFLAGS) $(NVLIBS)

# here are all the objects
GPUOBJS = integrals.o 

# make and compile
a.out:  $(GPUOBJS)
	$(NVCC) -o a.out $(GPUOBJS) 

integrals.o: integrals.cu
	$(NVCC) -c integrals.cu

clean:
	rm integrals.o

delete:
	touch a.out && rm a.out
