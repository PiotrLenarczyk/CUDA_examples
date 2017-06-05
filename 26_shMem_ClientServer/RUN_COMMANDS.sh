#!/bin/bash
touch shMemSrv && rm shMemSrv
g++ serverShMem.cpp -o shMemSrv && ./shMemSrv 5600
g++ client.c -o client #separate terminal: ./client localhost 5600



touch gpuClient && rm gpuClient
touch shMemPrintClient && rm shMemPrintClient

