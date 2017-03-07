#!/bin/bash
rm tester
`find /usr -name 'nvcc'` -Wno-deprecated-gpu-targets -O2 -c device.cu 
g++ -O2 -c host.cpp -I/usr/local/cuda/include/ 
`find /usr -name 'nvcc'` -Wno-deprecated-gpu-targets -o tester device.o host.o
rm *.o
clear 
./tester dataFile.txt 
