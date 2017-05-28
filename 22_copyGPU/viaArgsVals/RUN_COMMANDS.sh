#!/bin/bash 
touch a.out && rm a.out && qmake && make && clear 
cd childProc/ && make && cd ..
./a.out


