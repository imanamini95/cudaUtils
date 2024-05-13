//  nvcc -o ./src/medianBlur/medianBlur.dll --shared -Xcompiler -fPIC  ./src/medianBlur/medianBlur.cu
// nvcc -o ./src/medianBlur/medianBlur  ./src/medianBlur/medianBlur.cu
// ./src/medianBlur/medianBlur.exe

#include <stdio.h>

__global__ void medianBlur(int *input, int *output, int width, int height, int window_size)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height)
    {
        int half = window_size / 2;
        int count = 0;
        int window[9];

        for (int i = -half; i <= half; i++)
        {
            for (int j = -half; j <= half; j++)
            {
                int curr_col = min(max(col + j, 0), width - 1);
                int curr_row = min(max(row + i, 0), height - 1);
                window[count++] = input[curr_row * width + curr_col];
            }
        }

        for (int i = 0; i < window_size * window_size; i++)
        {
            for (int j = i + 1; j < window_size * window_size; j++)
            {
                if (window[i] > window[j])
                {
                    int temp = window[i];
                    window[i] = window[j];
                    window[j] = temp;
                }
            }
        }

        output[row * width + col] = window[(window_size * window_size) / 2];
    }
}

extern "C"
{
    __declspec(dllexport) void cuda_medianBlur(int *input, int *output, int width, int height, int window_size)
    {
        int *dev_input, *dev_output;
        size_t size = width * height * sizeof(int);

        cudaMalloc((void **)&dev_input, size);
        cudaMalloc((void **)&dev_output, size);

        cudaMemcpy(dev_input, input, size, cudaMemcpyHostToDevice);

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

        medianBlur<<<numBlocks, threadsPerBlock>>>(dev_input, dev_output, width, height, window_size);
        cudaDeviceSynchronize();

        cudaMemcpy(output, dev_output, size, cudaMemcpyDeviceToHost);

        cudaFree(dev_input);
        cudaFree(dev_output);
    }
}

int main()
{
    int width = 4;
    int height = 4;
    int window_size = 3;
    int input[16] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160};
    int output[16];

    cuda_medianBlur(input, output, width, height, window_size);

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            printf("%d ", output[i * width + j]);
        }
        printf("\n");
    }

    free(input);
    free(output);

    return 0;
}
