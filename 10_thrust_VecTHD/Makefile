# Build tools 
FLAGS = -Wno-deprecated-gpu-targets -std=c++11
NVCC = `find /usr -name 'nvcc'` $(FLAGS)

# here are all the objects
GPUOBJS = vecTHD.o 

# make and compile
a.out:  $(GPUOBJS)
	$(NVCC) -o a.out $(GPUOBJS) 

vecTHD.o: vecTHD.cu
	$(NVCC) -c vecTHD.cu

clean:
	rm vecTHD.o

delete:
	touch a.out && rm a.out
