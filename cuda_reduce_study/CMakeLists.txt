# ============ Hello World ============ #
add_executable(hello_world HelloWorld.cu)
target_link_libraries(hello_world 
    PRIVATE 
    CUDA::cudart      # 运行时库
    CUDA::cublas      # cublas库
)
# debug编译选项
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(hello_world PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

# ============ Vector Add baseline ============ #
add_executable(reduce_baseline ReduceBaseline.cu)
target_link_libraries(reduce_baseline 
    PRIVATE 
    CUDA::cudart      # 运行时库
    CUDA::cublas      # cublas库
)
# debug编译选项
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(reduce_baseline PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

# ============ Vector Add No Divergence ============ #
add_executable(reduce_no_divergence ReduceNoDivergence.cu)
target_link_libraries(reduce_no_divergence 
    PRIVATE 
    CUDA::cudart      # 运行时库
    CUDA::cublas      # cublas库
)
# debug编译选项
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(reduce_no_divergence PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

# ============ Vector Add No bank conflict ============ #
add_executable(reduce_no_bank_conflict ReduceNoBankConflict.cu)
target_link_libraries(reduce_no_bank_conflict 
    PRIVATE 
    CUDA::cudart      # 运行时库
    CUDA::cublas      # cublas库
)
# debug编译选项
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(reduce_no_bank_conflict PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

# ============ A thread add two nums ============ #
add_executable(reduce_add ReduceAdd.cu)
target_link_libraries(reduce_add 
    PRIVATE 
    CUDA::cudart      # 运行时库
    CUDA::cublas      # cublas库
)
# debug编译选项
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(reduce_add PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

# ============ Unroll Last Warp ============ #
add_executable(reduce_unroll_warp ReduceUnrollLastWarp.cu)
target_link_libraries(reduce_unroll_warp 
    PRIVATE 
    CUDA::cudart      # 运行时库
    CUDA::cublas      # cublas库
)
# debug编译选项
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(reduce_unroll_warp PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
