// nvcc -o ./Release/output/sum_array ./utils/sum_array.cu ./utils/common.cpp ./utils/cuda_common.cuh
// ./Release/output/sum_array.exe

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_common.cuh"

#include "common.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cstring>
#include <cmath>

__global__ void sumArrayGpu(int *a, int *b, int *c, int size)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < size)
    {
        c[gid] = a[gid] + b[gid];
    }
}

void sumArrayCpu(int *a, int *b, int *c, int size)
{
    for (int i = 0; i < size; i++)
    {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    int size = pow(2, 25);
    int blockSize = 512;

    int noBytes = size * sizeof(int);

    // host pointers
    int *hA, *hB, *gpuResults, *hC;
    hA = (int *)malloc(noBytes);
    hB = (int *)malloc(noBytes);
    hC = (int *)malloc(noBytes);
    gpuResults = (int *)malloc(noBytes);

    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++)
    {
        hA[i] = (int)(rand() & 0xff);
    }
    for (int i = 0; i < size; i++)
    {
        hB[i] = (int)(rand() & 0xff);
    }
    clock_t cpuStart, cpuEnd;
    cpuStart = clock();
    sumArrayCpu(hA, hB, hC, size);
    cpuEnd = clock();

    int *dA,
        *dB, *dC;
    gpuErrchk(cudaMalloc((int **)&dA, noBytes));
    gpuErrchk(cudaMalloc((int **)&dB, noBytes));
    gpuErrchk(cudaMalloc((int **)&dC, noBytes));

    clock_t hToDStart, hToDEnd;
    hToDStart = clock();
    cudaMemcpy(dA, hA, noBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, noBytes, cudaMemcpyHostToDevice);
    hToDEnd = clock();

    dim3 block(blockSize);
    dim3 grid((size / block.x) + 1);

    clock_t gpuStart, gpuEnd;
    gpuStart = clock();
    sumArrayGpu<<<grid, block>>>(dA, dB, dC, size);
    cudaDeviceSynchronize();
    gpuEnd = clock();

    clock_t dToHStart, dToHEnd;
    dToHStart = clock();
    cudaMemcpy(gpuResults, dC, noBytes, cudaMemcpyDeviceToHost);
    dToHEnd = clock();

    // array comparison
    compareArrays(hC, gpuResults, size);

    printf("Sum Array CPU execution time: %4.6f \n", (double)((double)(cpuEnd - cpuStart) / CLOCKS_PER_SEC));
    printf("Sum Array GPU execution time: %4.6f \n", (double)((double)(gpuEnd - gpuStart) / CLOCKS_PER_SEC));
    printf("Host to device transfer time: %4.6f \n", (double)((double)(hToDEnd - hToDStart) / CLOCKS_PER_SEC));
    printf("Device to host transfer time: %4.6f \n", (double)((double)(dToHEnd - dToHStart) / CLOCKS_PER_SEC));
    printf("Total GPU time: %4.6f \n", (double)((double)(dToHEnd - hToDStart) / CLOCKS_PER_SEC));

    cudaFree(dC);
    cudaFree(dB);
    cudaFree(dA);

    free(gpuResults);
    free(hA);
    free(hB);

    return 0;
}
