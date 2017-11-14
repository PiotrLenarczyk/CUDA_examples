# Build tools 						maximize calculations precision
FLAGS=-Wno-deprecated-gpu-targets -std=c++11 -O3 --fmad=true
DynParFLAGS = -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_61,code=sm_61 -rdc=true 
NVLIBS = -L /usr/lib/x86_64-linux-gnu/ -lcudadevrt
NVCC = `find /usr/local -name 'nvcc' -print -quit` $(FLAGS) $(DynParFLAGS) $(NVLIBS)

# here are all the objects
GPUOBJS = dct1d.o 

# make and compile
a.out:  $(GPUOBJS)
	$(NVCC) -o a.out $(GPUOBJS) 

dct1d.o: dct1d.cu
	$(NVCC) -c dct1d.cu

clean:
	rm dct1d.o

delete:
	touch a.out && rm a.out
