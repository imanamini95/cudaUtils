// nvcc -o ./Release/output/organization_of_threads ./utils/organization_of_threads.cu
// ./Release/output/organization_of_threads.exe

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void printThreadsIds()
{
    printf("blockIdx.x : %d, blockIdx.y : %d, blockIdx.z : %d, blockDim.x : %d, blockDim.y : %d, blockDim.z : %d, \n",
           blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z);
}

__global__ void printDetails()
{
    printf("threadIdx.x : %d, threadIdx.y : %d, threadIdx.z : %d, \n",
           threadIdx.x, threadIdx.y, threadIdx.z);
}

int main()
{
    int nx, ny;
    nx = 16;
    ny = 16;

    dim3 block(2, 2);
    dim3 grid(nx / block.x, ny / block.y);

    printThreadsIds<<<block, grid>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}
