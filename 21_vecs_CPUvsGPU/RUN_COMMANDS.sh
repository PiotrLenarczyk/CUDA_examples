#!/bin/bash
make delete && make -j`nproc` && make clean && clear && ./a.out 
rm *.out
