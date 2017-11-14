#!/bin/bash
#communication is unidirectional! from clients to listening server
touch shMemSrv && rm shMemSrv && touch gpuClient && rm gpuClient && touch printShMem && rm printShMem
g++ serverShMem.cpp -o shMemSrv 
`find /usr -type f -name 'nvcc'` -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_61,code=sm_61 -rdc=true --fmad=true -std=c++11 gpuClient.cu -o gpuClient
g++ printSERVERShMem.cpp -o printShMem 
./shMemSrv 5600 #now server is listening on port 5600 with SERVER_SHMID
#at new terminal send shared memory struct from client to server: #./gpuClient localhost 5600
#at new terminal one can view current server received shared memory struct from clients: #./printShMem SERVER_SHMID
#destroy server shared memory: #./printShMem SERVER_SHMID -d
