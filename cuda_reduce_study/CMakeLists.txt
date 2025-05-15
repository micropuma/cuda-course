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