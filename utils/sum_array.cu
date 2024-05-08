// nvcc -o sum_array ./Release/output/sum_array.cu common.cpp
// ./Release/output/sum_array.exe

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cstring>

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
    int size = 10000;
    int blockSize = 128;

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

    sumArrayCpu(hA, hB, hC, size);

    int *dA,
        *dB, *dC;
    cudaMalloc((int **)&dA, noBytes);
    cudaMalloc((int **)&dB, noBytes);
    cudaMalloc((int **)&dC, noBytes);

    cudaMemcpy(dA, hA, noBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, noBytes, cudaMemcpyHostToDevice);

    dim3 block(blockSize);
    dim3 grid((size / block.x) + 1);

    sumArrayGpu<<<grid, block>>>(dA, dB, dC, size);
    cudaDeviceSynchronize();

    cudaMemcpy(gpuResults, dC, noBytes, cudaMemcpyDeviceToHost);

    // array comparison
    compareArrays(hC, gpuResults, size);

    cudaFree(dC);
    cudaFree(dB);
    cudaFree(dA);

    free(gpuResults);
    free(hA);
    free(hB);

    return 0;
}
