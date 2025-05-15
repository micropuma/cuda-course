add_executable(my_reduce reduce.cu)
target_include_directories(my_reduce PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

# debug编译选项
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(my_reduce PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()