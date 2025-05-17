#!/bin/bash

if [ -d "build" ]; then
	rm -rf build
fi

mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j $(nproc)

cd bin
./reduce_baseline
./reduce_no_divergence
./reduce_no_bank_conflict
./reduce_add
./reduce_unroll_warp
./reduce_complete_unroll
./reduce_multi_add
./ReduceShuffle
