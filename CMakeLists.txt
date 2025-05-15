cmake_minimum_required(VERSION 3.30)
project(cuda_practice VERSION 0.1.0 LANGUAGES CXX C CUDA)

# 设置可执行文件输出目录（构建目录下的bin文件夹）
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(CUDAToolkit)
add_subdirectory(cuda_reduce_study)