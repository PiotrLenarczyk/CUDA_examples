# Build tools 
FLAGS=-Wno-deprecated-gpu-targets -std=c++11
DynParFLAGS = -arch=sm_35 -rdc=true 
# NVLIBS = -L /usr/lib/x86_64-linux-gnu/ -lcudadevrt -L /usr/local/cuda-8.0/lib64 -lcufft
# NVINC = -I /usr/local/cuda-8.0/include/thrust
NVLIBS = -L /usr/lib/x86_64-linux-gnu/ -lcudadevrt -lcufft
NVINC = -I /usr/include/thrust/
#NVCC = `find /usr -name 'nvcc' -print -quit` $(FLAGS) $(DynParFLAGS) $(NVLIBS)
NVCC = /usr/bin/nvcc $(FLAGS) $(DynParFLAGS) $(NVLIBS) $(NVINC)
# NVCC = /usr/local/cuda-8.0/bin/nvcc $(FLAGS) $(DynParFLAGS) $(NVLIBS) $(NVINC)

# here are all the objects
GPUOBJS = vecD_DCT.o 

# make and compile
a.out:  $(GPUOBJS)
	$(NVCC) -o a.out $(GPUOBJS) 

vecD_DCT.o: vecD_DCT.cu
	$(NVCC) -c vecD_DCT.cu

clean:
	rm vecD_DCT.o

delete:
	rm a.out
