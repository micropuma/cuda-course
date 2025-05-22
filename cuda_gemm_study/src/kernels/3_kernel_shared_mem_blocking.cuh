#pragma once

#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <cublas_v2.h>

/*

Matrix sizes:
MxK * KxN = MxN

*/

// 这是支持共享内存的矩阵乘法
template <unsigned int BLOCK_SIZE> 
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
    // 计算当前的thread block是第几行和列
    int block_row = blockIdx.x;
    int block_col = blockIdx.y;

    // 计算当前的thread在block中的行和列
    int thread_row = threadIdx.x / BLOCK_SIZE;
    int thread_col = threadIdx.x % BLOCK_SIZE;

    // 为A何B开辟共享内存，大小均为BLOCK_SIZE*BLOCK_SIZE
    __shared__ float A_shared[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float B_shared[BLOCK_SIZE * BLOCK_SIZE];

    // 首先计算A，B，C的起始位置
    A += block_row * BLOCK_SIZE * K;
    B += block_col * BLOCK_SIZE;
    C += block_row * BLOCK_SIZE * N + block_col * BLOCK_SIZE;
    
    // 当前的计算点用寄存器存储
    float temp = 0.0f;

    // 遍历每个blockSize的块
    for (int block_index = 0; block_index < K; block_index += BLOCK_SIZE) {
        // 每个线程加载一个元素到共享内存
        A_shared[thread_row * BLOCK_SIZE + thread_col] = A[thread_row * K + thread_col];
        B_shared[thread_row * BLOCK_SIZE + thread_col] = B[thread_row * N + thread_col];
        __syncthreads(); // 等待所有线程加载完毕

        // 每一轮迭代不断更新A，B的位置
        A += BLOCK_SIZE;
        B += BLOCK_SIZE * N;

        // 计算C的每个元素
        for (int i = 0; i < BLOCK_SIZE; i++) {
            // A按照列方向访问，B按照行方向访问
            temp += A_shared[thread_row * BLOCK_SIZE + i] * B_shared[i * BLOCK_SIZE + thread_col];
        }

        __syncthreads(); // 等待所有线程计算完毕
    }

    // C = α*(A@B)+β*C
    C[thread_row * N + thread_col] = alpha * temp + beta * C[thread_row * N + thread_col]; 
}