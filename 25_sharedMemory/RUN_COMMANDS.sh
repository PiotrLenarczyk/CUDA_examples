#!/bin/bash
qmake && make -j`nproc` && make clean && `find /usr -type f -name 'nvcc' ` gpuClient.cu -std=c++11 -arch=sm_50 -o aClient.out
clear && ./a.out && ./aClient.out
rm *.out Makefile
