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

template <unsigned int BLOCK_SIZE> 
__global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
    // 注意这里的线程索引变成一维度的了
    const uint x = blockIdx.x * BLOCK_SIZE + threadIdx.x / BLOCK_SIZE;
    const uint y = blockIdx.y * BLOCK_SIZE + threadIdx.x % BLOCK_SIZE;

    // 计算 C 矩阵的每个元素
    if (x < M && y < N) {     // 检查每个thread是否需要工作
        float temp = 0.0f;
        for (int i = 0; i < K; i++) {
            temp += A[x * K + i] * B[i * N + y];
        }

        // C = α*(A@B)+β*C
        C[x*N + y] = alpha * temp + beta * C[x*N + y];
    }
}