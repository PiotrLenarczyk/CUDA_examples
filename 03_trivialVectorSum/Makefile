# Build tools 
FLAGS = -Wno-deprecated-gpu-targets -std=c++11
NVCC = `find /usr -name 'nvcc'` $(FLAGS)

# here are all the objects
GPUOBJS = vecSum.o 

# make and compile
a.out:  $(GPUOBJS)
	$(NVCC) -o a.out $(GPUOBJS) 

vecSum.o: vecSum.cu
	$(NVCC) -c vecSum.cu

clean:
	rm vecSum.o

delete:
	rm a.out
