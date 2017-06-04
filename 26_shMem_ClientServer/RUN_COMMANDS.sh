#!/bin/bash
touch shMemSrv && rm shMemSrv
g++ serverShMem.cpp -o shMemSrv && ./shMemSrv
touch gpuClient && rm gpuClient

touch shMemPrintClient && rm shMemPrintClient

