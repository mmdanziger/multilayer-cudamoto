#!/bin/bash
mkdir -p ./cudamoto2-viewer/build && pushd ./cudamoto2-viewer/build
/usr/bin/qmake ../src
make
popd
 
