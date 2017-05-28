#!/bin/bash
rm a.out aChild.out & make clean && qmake && make && `find /usr/local -name 'nvcc'` -arch=sm_35 shVarChild.cu -o aChild.out && clear && ./a.out
 
