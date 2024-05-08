// nvcc -o ./Release/output/device_query ./utils/device_query.cu
// ./Release/output/device_query.exe

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

void queryDevice()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        printf("No CUDA supported device found");
    }

    int devNo = 0;
    cudaDeviceProp iProp;
    cudaGetDeviceProperties(&iProp, devNo);

    printf("Device %d: %s\n", devNo, iProp.name);
    printf("  Number of MPs: %d\n", iProp.multiProcessorCount);
    printf("  clock rate: %d\n", iProp.clockRate);
    printf("  compute capabilities: %d.%d\n", iProp.minor, iProp.major);
    printf("  Total global memory: %4.2f KB\n", iProp.totalGlobalMem / 1024.0);
}

int main()
{
    queryDevice();
    return 0;
}
