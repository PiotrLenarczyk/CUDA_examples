#!/bin/bash
make delete && make -j`nproc` && make clean && clear && for (( i=0; i<2; i++ ));do ./a.out;done
