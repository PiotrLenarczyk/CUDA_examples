#!/bin/bash
#also one can run: 
#                   make && ./a.out
make delete && touch ./bandwidthTest_CUDASamples/bandwidthTest && rm ./bandwidthTest_CUDASamples/bandwidthTest
make -j `nproc` && cd bandwidthTest_CUDASamples/; make -j `nproc`; cd ..
make clean && clear  && ./a.out
echo "NOTE P.S.: Rubik's Cube for single element 7D model of 6D space - each regular hexahedron is a single 3D space"
echo ""; echo "CUDA samples bandwidthTest:"; 
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
./bandwidthTest_CUDASamples/bandwidthTest
