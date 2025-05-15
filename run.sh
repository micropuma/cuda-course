#!/bin/bash

if [ -d "build" ]; then
	rm -rf build
fi

mkdir build
cd build
cmake ..
make -j $(nproc)

cd bin
./reduce_baseline
./reduce_no_divergence
