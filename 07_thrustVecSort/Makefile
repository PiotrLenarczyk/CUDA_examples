# Build tools 
FLAGS = -Wno-deprecated-gpu-targets -std=c++11
NVCC = `find /usr -name 'nvcc'` $(FLAGS)

# here are all the objects
GPUOBJS = thrustSort.o 

# make and compile
a.out:  $(GPUOBJS)
	$(NVCC) -o a.out $(GPUOBJS) 

thrustSort.o: thrustSort.cu
	$(NVCC) -c thrustSort.cu

clean:
	rm thrustSort.o

delete:
	rm a.out
