#include <bits/stdc++.h>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <sys/time.h>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"

# define THREAD_PER_BLOCK 256

// 核函数：规约函数
__global__ void reduce(float *device_input, float *device_output) {
    // 计算每个快的索引位置
    float *device_data = device_input + blockIdx.x * blockDim.x * 2;     // 每个block的起始位置

    // 支持shared memory
    volatile __shared__ float shared_data[THREAD_PER_BLOCK];      // 共享内存，注意这里volatile关键字的使用
    shared_data[threadIdx.x] = device_data[threadIdx.x] + device_data[threadIdx.x + blockDim.x]; // 将数据拷贝到共享内存
    __syncthreads();                                     // 同步线程，确保所有线程都完成数据拷贝
    
    // 这一版本支持no bank conflict，判断是否是最后一个warp了，就不用syncthreads了
    for (int i = blockDim.x / 2; i > 32; i /= 2) {    // i的跨度需要是blockDim.x/2启始
        // 只有前半区参与计算
        if (threadIdx.x < i) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + i]; // 规约
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
    int block_num = N / THREAD_PER_BLOCK / 2; // 计算块数
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
        for (int j = 0; j < 2*THREAD_PER_BLOCK; j++) {
            res[i] += host_input[i * 2 * THREAD_PER_BLOCK + j];
        }
    }

    //=========== GPU计算 ==============
    // 1. 将数据从主机拷贝到设备
    cudaMemcpy(device_input, host_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // 2. 启动核函数
    dim3 Grid(block_num, 1);       
    dim3 Block(THREAD_PER_BLOCK, 1);
    reduce<<<Grid, Block>>>(device_input, device_output);

    // 3. 将结果从设备拷贝到主机
    // 目前的reduce是每个block计算一个值，所以我们需要将每个block的结果拷贝到主机
    cudaMemcpy(host_output, device_output, block_num * sizeof(float), cudaMemcpyDeviceToHost);

    // 检测结果
    if (check(res, host_output, block_num)) {
        printf("================== Reduce Unroll Last Warp ==================\n");
        printf("Result is correct!\n");
    } else {
        printf("================== Reduce Unroll Last Warp ==================\n");
        printf("Result is incorrect!\n");
    }

    cudaFree(device_input);
    cudaFree(device_output);
    free(host_input);
    free(host_output);
    free(res);

    return 0;
}