# Build tools 
FLAGS = -Wno-deprecated-gpu-targets -std=c++11
NVCC = `find /usr -name 'nvcc'` $(FLAGS)

# here are all the objects
GPUOBJS = vecTuple.o

# make and compile
a.out:  $(GPUOBJS)
	$(NVCC) -o a.out $(GPUOBJS)

vecTuple.o: vecTuple.cu
	$(NVCC) -c vecTuple.cu

clean:
	rm vecTuple.o

delete:
	rm a.out
