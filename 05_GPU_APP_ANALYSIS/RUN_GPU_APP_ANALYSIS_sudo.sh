#!/bin/bash
#profiling for analysis: trivial Vector Sum
echo "======================================================"
make -j `nproc` -C ../03_trivialVectorSum
sudo `sudo find /usr/local -name 'nvprof'` ../03_trivialVectorSum/a.out
make clean -C ../03_trivialVectorSum && make delete -C ../03_trivialVectorSum
echo "======================================================"
echo ""
echo "======================================================"
#profiling for analysis: basic asynchronous optimalization for Vector Sum
make -j `nproc` -C ../04_asyncOptimalization
sudo `sudo find /usr/local -name 'nvprof'` ../04_asyncOptimalization/a.out
make clean -C ../04_asyncOptimalization && make delete -C ../04_asyncOptimalization
