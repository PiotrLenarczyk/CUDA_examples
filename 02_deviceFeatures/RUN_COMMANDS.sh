#!/bin/bash
#also one can run: 
#                   make && ./a.out
touch a.out && rm a.out
touch ./bandwidthTest_CUDASamples/bandwidthTest && rm ./bandwidthTest_CUDASamples/bandwidthTest
make delete
make -j `nproc`
cd bandwidthTest_CUDASamples/; make -j `nproc`; cd ..
make clean
clear 
./a.out
echo "NOTE P.S.: Rubik's Cube for single element 7D model of 6D space"
echo ""; echo "CUDA samples bandwidthTest:"; 
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
./bandwidthTest_CUDASamples/bandwidthTest
