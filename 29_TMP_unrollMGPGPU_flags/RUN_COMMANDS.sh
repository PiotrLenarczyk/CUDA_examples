#!/bin/bash
make delete && make -j`nproc` && make clean && clear && for (( i=0; i<2; i++ ));do echo "===="; echo "program run no. [$i]"; ./a.out; echo "====";done && touch a.out && rm a.out
