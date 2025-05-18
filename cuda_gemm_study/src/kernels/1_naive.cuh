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

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
    // 获取索引
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    // 计算 C 矩阵的每个元素
    if (x < N && y < M) {     // 检查每个thread是否需要工作
        float temp = 0.0f;
        for (int i = 0; i < K; i++) {
            temp += A[x * K + i] * B[i * N + y];
        }

        // C = α*(A@B)+β*C
        C[x*K + y] = alpha * temp + beta * C[x*K + y];
    }
}