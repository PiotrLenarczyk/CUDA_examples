#!/bin/bash
touch a.out && rm a.out
qmake && make -j`nproc` && make clean && ./a.out
`find /usr -type f -name 'nvcc' ` gpuClient.cu -std=c++11 -arch=sm_50 -o aClient.out && ./aClient.out
rm *.out Makefile
