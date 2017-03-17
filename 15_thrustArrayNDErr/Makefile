# Build tools 
FLAGS = -Wno-deprecated-gpu-targets -std=c++11
NVCC = /usr/bin/nvcc $(FLAGS)

# here are all the objects
GPUOBJS = array2D.o 

# make and compile
a.out:  $(GPUOBJS)
	$(NVCC) -o a.out $(GPUOBJS) 

array2D.o: array2D.cu
	$(NVCC) -c array2D.cu

clean:
	rm array2D.o

delete:
	rm a.out
