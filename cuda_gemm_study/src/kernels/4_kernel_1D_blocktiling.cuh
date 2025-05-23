#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm1DBlocktiling(int M, int N, int K, float alpha,
                                   const float *A, const float *B, float beta,
                                   float *C) {
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int threadRow = threadIdx.x / BN;
    int threadCol = threadIdx.x % BN;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += blockRow * BM * K;
    B += blockCol * BN;
    C += blockRow * BM * N + blockCol * BN;

    int innerRowA = threadIdx.x / BK;
    int innerColA = threadIdx.x % BK;
    int innerRowB = threadIdx.x / BN;
    int innerColB = threadIdx.x % BN;

    float threadResults[TM] = {0.0};

    for (int i = 0; i < K; i += BK) {
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
        __syncthreads();

        for (int j = 0; j < BK; ++j) {
            float temp = Bs[j * BN + threadCol];
            for (int t = 0; t < TM; ++t) {
                threadResults[t] += As[(threadRow * TM + t) * BK + j] * temp;
            }
        }
        __syncthreads();

        A += BK;         // 沿A的列方向移动BK
        B += BK * N;     // 沿B的行方向移动BK（每行长度N）
    }

    // 统一写回结果
    for (int resIdx = 0; resIdx < TM; ++resIdx) {
        C[(threadRow * TM + resIdx) * N + threadCol] =
            alpha * threadResults[resIdx] + beta * C[(threadRow * TM + resIdx) * N + threadCol];
    }
}