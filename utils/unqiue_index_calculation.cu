// nvcc -o ./Release/output/unqiue_index_calculation ./utils/unqiue_index_calculation.cu
// ./Release/output/unqiue_index_calculation.exe

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void uniqueIdxCalcThreadIdx(int *input)
{
    int tid = threadIdx.x;
    printf("threadIdx: %d, value: %d \n", tid, input[tid]);
}

__global__ void uniqueGidCalc(int *input)
{
    int tid = threadIdx.x;
    int offset = blockIdx.x * blockDim.x;
    int gid = tid + offset;
    printf("blockIdx.x: %d, threadIdx.x: %d, gid: %d , value: %d  \n", blockIdx.x, threadIdx.x, gid, input[gid]);
}

__global__ void uniqueGidCalc2D2D(int *input)
{
    int tid = blockDim.x * threadIdx.y + threadIdx.x;

    int numThreadInBlock = blockDim.x * blockDim.y;
    int blockOffset = blockIdx.x * numThreadInBlock;

    int numThreadInRow = numThreadInBlock * gridDim.x;
    int rowOffset = numThreadInRow * blockIdx.y;

    int gid = tid + blockOffset + rowOffset;
    printf("blockIdx.x: %d, threadIdx.x: %d, gid: %d , value: %d  \n", blockIdx.x, threadIdx.x, gid, input[gid]);
}

int main()
{
    int arraySize = 16;
    int arrayByteSize = sizeof(int) * arraySize;
    int hData[] = {10, 15, 25, 54, 89, 15, 2, 66, 17, 47, 89, 23, 15, 19, 72, 26};

    for (int i = 0; i < arraySize; i++)
    {
        printf("%d ", hData[i]);
    }

    int *dData;
    cudaMalloc((void **)&dData, arrayByteSize);
    cudaMemcpy(dData, hData, arrayByteSize, cudaMemcpyHostToDevice);

    dim3 block(2, 2);
    dim3 grid(2, 2);

    uniqueGidCalc2D2D<<<grid, block>>>(dData);
    cudaDeviceSynchronize();
    cudaDeviceReset();

    return 0;
}
