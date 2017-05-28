#!/bin/bash
touch a.out && rm a.out && qmake && make -j`nproc` && make clean
cd gpuCode/ && make && make clean && cd ..
./a.out && cd gpuCode/ && ./a.out
