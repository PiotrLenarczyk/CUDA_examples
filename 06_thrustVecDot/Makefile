# Build tools 
FLAGS = -Wno-deprecated-gpu-targets -std=c++11
NVCC = `find /usr/bin/ -name 'nvcc'` $(FLAGS)

# here are all the objects
GPUOBJS = thrustVecSum.o 

# make and compile
a.out:  $(GPUOBJS)
	$(NVCC) -o a.out $(GPUOBJS) 

thrustVecSum.o: thrustVecSum.cu
	$(NVCC) -c thrustVecSum.cu

clean:
	rm thrustVecSum.o

delete:
	rm a.out
