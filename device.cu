#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int device_id = 0;
    cudaSetDevice(device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    // 固定参数（不同架构可能不同，以 Ampere 为例）
    const int WARP_SIZE = 32;
    const int WARP_ALLOC_GRANULARITY = 4;  // Ampere架构经测试验证
    const int REG_ALLOC_UNIT_SIZE = 256;   // 每个寄存器分配单元大小 (参考 PTX 文档)
    const int CUDA_SHARED_MEM_OVERHEAD = 1024; // 运行时为每个块保留的共享内存开销

    // 打印硬件参数
    printf("| Metric\t\t\t| Value\t\t|\n");
    printf("|--------------------------------|---------------|\n");
    printf("| Name\t\t\t\t| %s\t|\n", prop.name);
    printf("| Compute Capability\t\t| %d.%d\t\t|\n", prop.major, prop.minor);
    printf("| max threads per block\t\t| %d\t\t|\n", prop.maxThreadsPerBlock);
    printf("| max threads per multiprocessor | %d\t\t|\n", prop.maxThreadsPerMultiProcessor);
    printf("| threads per warp\t\t| %d\t\t|\n", WARP_SIZE);
    printf("| warp allocation granularity\t| %d\t\t|\n", WARP_ALLOC_GRANULARITY);
    printf("| max regs per block\t\t| %d\t|\n", prop.regsPerBlock);
    printf("| max regs per multiprocessor\t| %d\t|\n", prop.regsPerMultiprocessor);
    printf("| reg allocation unit size\t| %d\t\t|\n", REG_ALLOC_UNIT_SIZE);
    printf("| reg allocation granularity\t| warp\t\t|\n");
    printf("| total global mem\t\t| %zu MB\t|\n", prop.totalGlobalMem / (1024 * 1024));
    printf("| max shared mem per block\t| %zu KB\t|\n", prop.sharedMemPerBlock / 1024);
    printf("| CUDA shared mem overhead (per block) | %d B\t|\n", CUDA_SHARED_MEM_OVERHEAD);
    printf("| shared mem per multiprocessor\t| %zu B\t|\n", prop.sharedMemPerMultiprocessor);
    printf("| multiprocessor count\t\t| %d\t\t|\n", prop.multiProcessorCount);

    return 0;
}