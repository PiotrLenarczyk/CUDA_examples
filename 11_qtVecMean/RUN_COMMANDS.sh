#!/bin/bash
#also one can run: 
#                   make && ./a.out
touch a.out && rm a.out
qmake
make -j `nproc`
make clean
clear 
./a.out
