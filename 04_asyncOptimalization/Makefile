# Build tools 
FLAGS = -Wno-deprecated-gpu-targets -std=c++11
NVCC = /usr/bin/nvcc $(FLAGS)

# here are all the objects
GPUOBJS = asyncOpto.o 

# make and compile
a.out:  $(GPUOBJS)
	$(NVCC) -o a.out $(GPUOBJS) 

asyncOpto.o: asyncOpto.cu
	$(NVCC) -c asyncOpto.cu

clean:
	rm asyncOpto.o

delete:
	rm a.out
