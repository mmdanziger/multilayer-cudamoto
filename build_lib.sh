#!/bin/bash
mkdir -p ./cudamoto2/build && pushd ./cudamoto2/build
nvcc  -O3 -DCOS_FACTOR  -Xcompiler="-fPIC" -std=c++11 -x cu -I. -dc ../src/Cudamoto2.cu -o Cudamoto2.o
nvcc -std=c++11 -O3  -DCOS_FACTOR  -dlink  -Xcompiler="-fPIC" -o libCudamoto2.o Cudamoto2.o -lcudart
ar rc libCudamoto2.so libCudamoto2.o Cudamoto2.o
ranlib libCudamoto2.so
cp libCudamoto2.so ../../cudamoto2-viewer/src
cp libCudamoto2.so ../../mains/
popd
