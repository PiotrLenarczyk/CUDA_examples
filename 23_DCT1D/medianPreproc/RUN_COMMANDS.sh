#!/bin/bash
make delete && make -j`nproc` && make clean && clear && time ./a.out

