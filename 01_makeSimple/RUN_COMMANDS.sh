#!/bin/bash
#also one can run: 
#                   make && ./a.out
make -j `nproc`
make clean
clear 
./a.out

