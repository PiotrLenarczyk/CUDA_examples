#!/bin/bash
make delete && make -j `nproc` && make clean && clear 
#first manner via bash commands ( mainly usable on single worker )
for (( i = 0; i < 4; i++ ))
do
    ./a.out $i &
done
exec bash &

# #GNU parallel as far as free ( more widely usable ):
touch commands.txt && rm commands.txt
for (( i = 0; i < 4; i++ )) #most of GPU workloads must fit in GPU scanty memory and possible Hyper-Q OS threads ( up to ~8 )
do
    echo "./a.out $i" &>> commands.txt
done
parallel -j`nproc` < commands.txt 
 
