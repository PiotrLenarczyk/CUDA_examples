# Build tools 						maximize calculations precision
FLAGS=-Wno-deprecated-gpu-targets -std=c++11 -O3 --fmad=true
DynParFLAGS = -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_61,code=sm_61 -rdc=true 
NVLIBS = -L /usr/lib/x86_64-linux-gnu/ -lcudadevrt
NVCC = `find /usr/local -name 'nvcc' -print -quit` $(FLAGS) $(DynParFLAGS) $(NVLIBS)

# here are all the objects
GPUOBJS = fp4.o 

# make and compile
a.out:  $(GPUOBJS)
	$(NVCC) -o a.out $(GPUOBJS) 

fp4.o: fp4.cu
	$(NVCC) -c fp4.cu

clean:
	rm fp4.o

delete:
	touch a.out && rm a.out
