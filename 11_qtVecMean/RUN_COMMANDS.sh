#!/bin/bash
#also one can run: 
#                   make && ./a.out
echo "some cuda-8.0 error..."
touch a.out && rm a.out && qmake && make -j `nproc` && make clean && clear && ./a.out
