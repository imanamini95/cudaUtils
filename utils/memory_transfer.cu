// nvcc -o ./Release/output/memory_transfer ./utils/memory_transfer.cu
// ./Release/output/memory_transfer.exe

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <time.h>

__global__ void memTrsTest(int *input)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("tid: %d, gid: %d, value: %d \n", threadIdx.x, gid, input[gid]);
}

int main()
{
    int size = 128;

    int byteSize = size * sizeof(int);

    int *hInput;

    hInput = (int *)malloc(byteSize);
    time_t t;
    srand((unsigned)time(&t));

    for (int i = 0; i < size; i++)
    {
        hInput[i] = (int)(rand() & 0xff);
    }

    int *dInput;
    cudaMalloc((void **)&dInput, byteSize);
    cudaMemcpy(dInput, hInput, byteSize, cudaMemcpyHostToDevice);

    dim3 block(64);
    dim3 grid(2);

    memTrsTest<<<grid, block>>>(dInput);
    cudaDeviceSynchronize();

    cudaDeviceReset();
}
