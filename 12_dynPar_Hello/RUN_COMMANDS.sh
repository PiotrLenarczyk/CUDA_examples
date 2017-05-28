#!/bin/bash
#also one can run: 
#                   make && ./a.out
echo "please enter you device CUDA capability - it could be found in device features"
touch a.out && rm a.out && make -j`nproc` && make clean && clear && ./a.out
