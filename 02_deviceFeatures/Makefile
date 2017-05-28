# Build tools 
NVCC = `find /usr -name 'nvcc' -print -quit` -Wno-deprecated-gpu-targets 

# here are all the objects
GPUOBJS = deviceFeatures.o 

# make and compile
a.out:  $(GPUOBJS)
	$(NVCC) -o a.out $(GPUOBJS) 

deviceFeatures.o: deviceFeatures.cu
	$(NVCC) -c deviceFeatures.cu

clean:
	rm deviceFeatures.o

delete:
	touch a.out && rm a.out
