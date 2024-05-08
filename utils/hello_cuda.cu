// build nvcc -o hello_cuda ./Release/output/hello_cuda.cu
// run ./Release/output/hello_cuda.exe

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void helloCuda()
{
    printf("Hello CUDA world \n");
}

int main()
{
    int nx, ny;
    nx = 16;
    ny = 4;

    dim3 block(8, 2);
    dim3 grid(nx / block.x, ny / block.y);
    helloCuda<<<block, grid>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}
