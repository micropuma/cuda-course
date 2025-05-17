cmake_minimum_required(VERSION 3.18)
project(cuda_reduce_study LANGUAGES CUDA CXX)

# Debug 模式下添加 CUDA 调试选项
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Enabling CUDA debug flags: -G -O0 -g")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -O0 -g")
endif()

# ============ Hello World ============ #
add_executable(hello_world HelloWorld.cu)
target_link_libraries(hello_world PRIVATE CUDA::cudart CUDA::cublas)

# ============ Vector Add baseline ============ #
add_executable(reduce_baseline ReduceBaseline.cu)
target_link_libraries(reduce_baseline PRIVATE CUDA::cudart CUDA::cublas)

# ============ Vector Add No Divergence ============ #
add_executable(reduce_no_divergence ReduceNoDivergence.cu)
target_link_libraries(reduce_no_divergence PRIVATE CUDA::cudart CUDA::cublas)

# ============ Vector Add No bank conflict ============ #
add_executable(reduce_no_bank_conflict ReduceNoBankConflict.cu)
target_link_libraries(reduce_no_bank_conflict PRIVATE CUDA::cudart CUDA::cublas)

# ============ A thread add two nums ============ #
add_executable(reduce_add ReduceAdd.cu)
target_link_libraries(reduce_add PRIVATE CUDA::cudart CUDA::cublas)

# ============ Unroll Last Warp ============ #
add_executable(reduce_unroll_warp ReduceUnrollLastWarp.cu)
target_link_libraries(reduce_unroll_warp PRIVATE CUDA::cudart CUDA::cublas)

# ============ Completely Unroll ============ #
add_executable(reduce_complete_unroll ReduceCompleteUnroll.cu)
target_link_libraries(reduce_complete_unroll PRIVATE CUDA::cudart CUDA::cublas)

# ============ Multi Add Version ============ #
add_executable(reduce_multi_add ReduceMultiAdd.cu)
target_link_libraries(reduce_multi_add PRIVATE CUDA::cudart CUDA::cublas)

# ============ Final Version With Shuffle enabled ============ #
add_executable(ReduceShuffle ReduceShuffle.cu)
target_link_libraries(ReduceShuffle PRIVATE CUDA::cudart CUDA::cublas)
