// nvcc -o ./Release/output/matrix_multiplication ./utils/matrix_multiplication.cu
// ./Release/output/matrix_multiplication.exe

#include <stdio.h>

#define N 3

__global__ void matrixMul(int *a, int *b, int *c)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if (row < N && col < N)
    {
        for (int i = 0; i < N; i++)
            sum += a[row * N + i] * b[i * N + col];
        c[row * N + col] = sum;
    }
}

int main()
{
    int a[N][N], b[N][N], c[N][N];
    int *d_a, *d_b, *d_c;

    // Initialize matrices a and b
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            a[i][j] = i + j;
            b[i][j] = i - j;
        }
    }

    // Allocate device memory
    cudaMalloc((void **)&d_a, N * N * sizeof(int));
    cudaMalloc((void **)&d_b, N * N * sizeof(int));
    cudaMalloc((void **)&d_c, N * N * sizeof(int));

    // Copy data to device memory
    cudaMemcpy(d_a, a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 grid(1, 1);
    dim3 block(N, N);

    // Launch kernel
    matrixMul<<<grid, block>>>(d_a, d_b, d_c);

    // Copy result back to host
    cudaMemcpy(c, d_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Display result
    printf("Matrix A:\n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%d\t", a[i][j]);
        }
        printf("\n");
    }

    printf("\nMatrix B:\n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%d\t", b[i][j]);
        }
        printf("\n");
    }

    printf("\nMatrix C:\n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%d\t", c[i][j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
