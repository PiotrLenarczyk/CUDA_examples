# Build tools g++-5 or g++-6
CXX = g++-5
NVCC = `find /usr -name 'nvcc'` -Wno-deprecated-gpu-targets 

# here are all the objects
GPUOBJS = cuexample.o 
OBJS = cppexample.o

# make and compile
a.out: $(OBJS) $(GPUOBJS)
	$(NVCC) -o a.out $(OBJS) $(GPUOBJS) 

cuexample.o: cuexample.cu
	$(NVCC) -c cuexample.cu

cppexample.o: cppexample.cpp
	$(CXX) -c cppexample.cpp

clean:
	rm cppexample.o cuexample.o

delete:
	rm a.out
