// nvcc -G -o ./Release/output/warp_divergence ./utils/warp_divergence.cu
// nvprof --metrics branch_efficiency ./Release/output/warp_divergence.exe

// experiment only works for GPU capability < 7.5 for higher Nsight

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_common.cuh"

__global__ void codeWithoutDivergence()
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    float a, b;
    a = b = 0;

    int warp_id = gid / 32;

    if (warp_id % 2 == 0)
    {
        a = 100.0;
        b = 50.0;
    }
    else
    {
        a = 200;
        b = 75;
    }
}

__global__ void divergenceCode()
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    float a, b;
    a = b = 0;

    if (gid % 2 == 0)
    {
        a = 100.0;
        b = 50.0;
    }
    else
    {
        a = 200;
        b = 75;
    }
}

int main(int argc, char **argv)
{
    printf("\n-----------------------WARP DIVERGENCE EXAMPLE------------------------ \n\n");

    int size = 1 << 22;

    dim3 block_size(128);
    dim3 grid_size((size + block_size.x - 1) / block_size.x);

    codeWithoutDivergence<<<grid_size, block_size>>>();
    cudaDeviceSynchronize();

    divergenceCode<<<grid_size, block_size>>>();
    cudaDeviceSynchronize();

    cudaDeviceReset();
    return 0;
}
