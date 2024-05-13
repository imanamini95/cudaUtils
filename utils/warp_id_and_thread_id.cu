// nvcc -o ./Release/output/warp_id_and_thread_id ./utils/warp_id_and_thread_id.cu
// ./Release/output/warp_id_and_thread_id.exe

#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void printDetailsOfWarps()
{
    int gid = blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

    int warp_id = threadIdx.x / 32;

    int gbid = blockIdx.y * gridDim.x + blockIdx.x;

    printf("tid : %d, bid.x : %d, bid.y : %d, gid : %d, warp_id : %d, gbid : %d \n",
           threadIdx.x, blockIdx.x, blockIdx.y, gid, warp_id, gbid);
}

int main(int argc, char **argv)
{
    dim3 block_size(42);
    dim3 grid_size(2, 2);

    printDetailsOfWarps<<<grid_size, block_size>>>();
    cudaDeviceSynchronize();

    cudaDeviceReset();
    return EXIT_SUCCESS;
}
