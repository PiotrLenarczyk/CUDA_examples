# Build tools 
FLAGS=-Wno-deprecated-gpu-targets -std=c++11
DynParFLAGS = -arch=sm_35 -rdc=true 
# NVLIBS = -L /usr/lib/x86_64-linux-gnu/ -lcudadevrt -L /usr/local/cuda-8.0/lib64
# NVINC = -I /usr/local/cuda-8.0/include/thrust
# NVCC = /usr/local/cuda-8.0/bin/nvcc $(FLAGS) $(DynParFLAGS) $(NVLIBS) $(NVINC)

NVLIBS = -L /usr/lib/x86_64-linux-gnu/ -lcudadevrt
NVINC = -I /usr/include/thrust/ /usr/local/cuda-8.0/targets/x86_64-linux/include/thrust/
# NVCC = /usr/bin/nvcc -ftz=true -prec-div=false -prec-sqrt=false $(FLAGS) $(DynParFLAGS) $(NVLIBS) $(NVINC)
#NVCC = /usr/bin/nvcc $(FLAGS) $(DynParFLAGS) $(NVLIBS) $(NVINC)

NVCC = `find /usr -name 'nvcc' -print -quit` $(FLAGS) $(DynParFLAGS) $(NVLIBS)

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
