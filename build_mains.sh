#!/bin/bash

mkdir -p ./mains/build && pushd ./mains/build
g++ -std=c++11 -O3 -I/usr/include/hdf5/serial -I/usr/local/cuda/include/ -I/usr/local/cuda/samples/common/inc -c ../main_singlerun.cpp -o main.o
g++ -std=c++11 -O3 -o main_single_run.bin main.o -L../ -lCudamoto2 -L/usr/local/cuda/lib64 -lcudart -L /usr/lib/x86_64-linux-gnu/hdf5/serial/ -lhdf5_cpp -lhdf5
popd
 
