# Build tools 
FLAGS = -Wno-deprecated-gpu-targets -std=c++11 
NVCC = `find /usr -name 'nvcc'` $(FLAGS)

# here are all the objects
GPUOBJS = parCUDA.o

# make and compile
a.out:  $(GPUOBJS)
	$(NVCC) -o a.out $(GPUOBJS) 

parCUDA.o: parCUDA.cu
	$(NVCC) -c parCUDA.cu

clean:
	rm parCUDA.o

delete:
	touch a.out && rm a.out
