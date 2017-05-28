# Build tools 
FLAGS=-Wno-deprecated-gpu-targets -std=c++11
DynParFLAGS = -arch=sm_35 -rdc=true 
NVLIBS = -L /usr/lib/x86_64-linux-gnu/ -L /usr/local/cuda-8.0/lib64 -lcufft
NVCC = `find /usr/local -name 'nvcc' -print -quit` $(FLAGS) $(DynParFLAGS) $(NVLIBS)

# here are all the objects
GPUOBJS = thrustCuFFT.o 

# make and compile
a.out:  $(GPUOBJS)
	$(NVCC) -o a.out $(GPUOBJS) 

thrustCuFFT.o: thrustCuFFT.cu
	$(NVCC) -c thrustCuFFT.cu

clean:
	rm thrustCuFFT.o

delete:
	touch a.out && rm a.out
