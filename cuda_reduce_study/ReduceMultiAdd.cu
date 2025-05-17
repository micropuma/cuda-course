#include <bits/stdc++.h>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <sys/time.h>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"

# define THREAD_PER_BLOCK 256

// 核函数：规约函数，支持multi add，使用归约
template <unsigned int BLOCK_SIZE, int TASK_PER_THREAD>
__global__ void reduce(float *device_input, float *device_output) {
    // 计算每个快的索引位置
    float *device_data = device_input + blockIdx.x * TASK_PER_THREAD * BLOCK_SIZE;     // 每个block的起始位置，计算逻辑是一个thread可以处理多少任务，一个block有多少个thread
    // 由此计算出一个block可以计算多少任务，每次位移这么多

    // 支持shared memory
    volatile __shared__ float shared_data[BLOCK_SIZE];      // 共享内存，注意这里volatile关键字的使用
    shared_data[threadIdx.x] = 0.0; // 初始化共享内存
    // 这里需要循环，做multiAdd
    for (int  i = 0; i < TASK_PER_THREAD; i++) {
        shared_data[threadIdx.x] += device_data[i * BLOCK_SIZE + threadIdx.x]; // 每个线程计算自己的数据
    }
    __syncthreads();                                     // 同步线程，确保所有线程都完成数据拷贝
    
    // 这一版本支持no bank conflict，判断是否是最后一个warp了，就不用syncthreads了
// #pragma unroll
//     for (int i = blockDim.x / 2; i > 32; i /= 2) {    // i的跨度需要是blockDim.x/2启始
//         // 只有前半区参与计算
//         if (threadIdx.x < i) {
//             shared_data[threadIdx.x] += shared_data[threadIdx.x + i]; // 规约
//         }

    if (BLOCK_SIZE >= 256) {
        if (threadIdx.x < 128) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + 128]; // 规约
        }
        __syncthreads(); // 同步线程，确保所有线程都完成数据规约
    }

    if (BLOCK_SIZE >= 128) {
        if (threadIdx.x < 64) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + 64]; // 规约
        }
        __syncthreads(); // 同步线程，确保所有线程都完成数据规约
    }

    // 归约最后一个warp的结果
    if (threadIdx.x < 32) { 
        // 规约最后一个warp的结果，这里做了展开，没必要在用if判断哪个thread参与计算，都直接计算，写起来更简单
        shared_data[threadIdx.x] += shared_data[threadIdx.x + 32];
        shared_data[threadIdx.x] += shared_data[threadIdx.x + 16];
        shared_data[threadIdx.x] += shared_data[threadIdx.x + 8];
        shared_data[threadIdx.x] += shared_data[threadIdx.x + 4];
        shared_data[threadIdx.x] += shared_data[threadIdx.x + 2];
        shared_data[threadIdx.x] += shared_data[threadIdx.x + 1];
    }

    // 只保留第一个线程的结果
    if (threadIdx.x == 0) {
        device_output[blockIdx.x] = shared_data[0]; // 将结果写入输出
    }
}

// 检查结果正确性
bool check(float *out, float *res, int n) {
    for (int i = 0; i < n; i++) {
        if (fabs(out[i] - res[i]) > 0.05) {       // 我们允许的误差范围
            printf("Error: out[%d] = %f, res[%d] = %f\n", i, out[i], i, res[i]);
            return false;
        }
    }
    return true;
}

int main() {
    const int N = 32 * 1024 * 1024; // 总计算量

    //========== 输入内存分配 ==============
    float *host_input = (float *)malloc(N * sizeof(float));       // 主机输入
    float *device_input; // 设备输入
    cudaMalloc((void **)&device_input, N * sizeof(float)); // 设备输入，注意传入二级指针

    //========== 输出内存分配 ==============
    constexpr int block_num = 1024; // 指定block num是1024
    constexpr int task_per_block = N / block_num; // 每个block的任务数
    constexpr int task_per_thread = task_per_block / THREAD_PER_BLOCK; // 每个线程的任务数

    float *host_output = (float *)malloc(block_num * sizeof(float));       // 主机输出
    float *device_output; // 设备输出
    cudaMalloc((void **)&device_output, block_num * sizeof(float)); // 设备输出，注意传入二级指针
    float *res = (float *)malloc(block_num * sizeof(float));                // 结果

    //========== 输入数据初始化 ==============
    for (int i = 0; i < N; i++) {
        // 生成随机数l
        host_input[i] = 2.0 * (float)drand48() - 1.0; // [-1, 1]之间的随机数
    }

    //========== CPU计算 ==============
    for (int i = 0; i < block_num; i++) {
        res[i] = 0.0;
        for (int j = 0; j < task_per_block; j++) {
            res[i] += host_input[i * task_per_block + j];
        }
    }

    //=========== GPU计算 ==============
    // 1. 将数据从主机拷贝到设备
    cudaMemcpy(device_input, host_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // 2. 启动核函数
    dim3 Grid(block_num, 1);       
    dim3 Block(THREAD_PER_BLOCK, 1);
    reduce<THREAD_PER_BLOCK, task_per_thread><<<Grid, Block>>>(device_input, device_output);

    // 3. 将结果从设备拷贝到主机
    // 目前的reduce是每个block计算一个值，所以我们需要将每个block的结果拷贝到主机
    cudaMemcpy(host_output, device_output, block_num * sizeof(float), cudaMemcpyDeviceToHost);

    // 检测结果
    if (check(res, host_output, block_num)) {
        printf("================== Reduce Multi Add ==================\n");
        printf("Result is correct!\n");
    } else {
        printf("================== Reduce Multi Add ==================\n");
        printf("Result is incorrect!\n");
    }

    cudaFree(device_input);
    cudaFree(device_output);
    free(host_input);
    free(host_output);
    free(res);

    return 0;
}