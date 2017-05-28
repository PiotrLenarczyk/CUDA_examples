#!/bin/bash
#also one can run: 
#   make && ./a.out
make delete && make -j `nproc` && make clean && clear && ./a.out

