cmake_minimum_required(VERSION 3.18)
project(cuda_gemm_study LANGUAGES CUDA CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)                # 或者选择显示的在cmake 命令中添加 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

find_package(CUDA REQUIRED)

# 确保cuda是存在的
include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CUDA_COMPUTE_CAPABILITY 86)

# 在debug mode中做如下设置，会在编译时生成ptx文件，并且开启-G选项，影响性能
add_compile_options("$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")

# Configure header file search paths
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)
# 收集所有src下的文件
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)

# sgemm可执行，完成矩阵乘法运算
add_executable(sgemm sgemm.cu ${SRC})
set_target_properties(sgemm PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(sgemm ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

