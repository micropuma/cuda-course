cmake_minimum_required(VERSION 3.30)
project(cuda_practice VERSION 0.1.0 LANGUAGES CXX C CUDA)
find_package(CUDAToolkit)
add_subdirectory(cuda_reduce_study)