# Build tools 
FLAGS=-Wno-deprecated-gpu-targets -std=c++11 -O3
DynParFLAGS = -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_61,code=sm_61 -rdc=true 
NVLIBS = -L /usr/lib/x86_64-linux-gnu/ -lcudadevrt
NVINC = -I /usr/include/thrust/ -I /usr/local/cuda-8.0/targets/x86_64-linux/include/thrust/ 
NVCC = `find /usr/local -name 'nvcc' -print -quit` $(FLAGS) $(DynParFLAGS) $(NVLIBS) $(NVINC)

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
	touch a.out && rm a.out
