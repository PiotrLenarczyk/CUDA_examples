#!/bin/bash
#also one can run: 
#                   make && ./a.out
locPath=`pwd`
echo $locPath
make delete && touch ./bandwidthTest_CUDASamples/bandwidthTest && rm ./bandwidthTest_CUDASamples/bandwidthTest
make -j `nproc` && cd bandwidthTest_CUDASamples/; make -j `nproc`; cd ..
clear
$locPath/a.out && echo "NOTE P.S.: Rubik's Cube for single element 7D model of 6D space - each regular hexahedron is a single 3D space"
rm $locPath/a.out $locPath/deviceFeatures.o;
$locPath/CUDA-Z-0.10.251-64bit.run &>> /tmp/logCudaZ.txt; rm /tmp/logCudaZ.txt

