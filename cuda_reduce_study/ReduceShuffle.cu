#include <bits/stdc++.h>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <sys/time.h>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"

# define THREAD_PER_BLOCK 256
#define FULL_MASK 0xffffffff

// 核函数：规约函数，支持shuffle的版本
template <unsigned int BLOCK_SIZE, int TASK_PER_THREAD>
__global__ void reduce(float *device_input, float *device_output) {
    // 计算每个快的索引位置
    float *device_data = device_input + blockIdx.x * TASK_PER_THREAD * BLOCK_SIZE;     // 每个block的起始位置，计算逻辑是一个thread可以处理多少任务，一个block有多少个thread
    // 由此计算出一个block可以计算多少任务，每次位移这么多

    // 用register
    float sum = 0.0;
    // 这里需要循环，做multiAdd
    for (int  i = 0; i < TASK_PER_THREAD; i++) {
        sum += device_data[i * BLOCK_SIZE + threadIdx.x]; // 每个线程计算自己的数据
    }
   
    // 一个warp的计算，利用shuffle instruction
    // 具体参考https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(FULL_MASK, sum, offset); // 规约
    } 

    // 在一个warp内归约好后，还要汇总各个warp的结果。由于GPU最多1024个线程，所以这里的warp数不会超过32，所以最多两层warp内reduce
    // 这里需要用shared memory来缓存结果
    __shared__ float shared_data[32];
    unsigned int lane_id = threadIdx.x % warpSize; // 计算lane id
    unsigned int warp_id = threadIdx.x / warpSize; // 计算warp id
    // 只有每个warp的第一个lane参与计算
    if (lane_id == 0) {
        shared_data[warp_id] = sum; // 将结果写入共享内存
    }
    __syncthreads(); // 同步线程，确保所有线程(按照warp执行)都完成数据写入

    // 只保留第一个线程的结果
    if (threadIdx.x == 0) {
        device_output[blockIdx.x] = sum; // 将结果写入输出
    }

    // 继续第二轮的reduction
    if (warp_id == 0) {
        // 从共享内存读入有个trick，就是可能32个缓存中，后面的warp没有写入数据，均是零
        // 加入有128个线程per block，则warp有4个，所以到了第二轮，只有前4个需要计算，后面的共享内存全为零
        sum = (lane_id < (blockDim.x / 32)) ? shared_data[lane_id] : 0.0; // 读取共享内存中的结果\

        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(FULL_MASK, sum, offset); // 规约
        } 
    }

    if (threadIdx.x == 0) {
        device_output[blockIdx.x] = sum; // 将结果写入输出
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
        printf("================== Reduce Shuffle ==================\n");
        printf("Result is correct!\n");
    } else {
        printf("================== Reduce Shuffle ==================\n");
        printf("Result is incorrect!\n");
    }

    cudaFree(device_input);
    cudaFree(device_output);
    free(host_input);
    free(host_output);
    free(res);

    return 0;
}