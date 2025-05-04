#include <iostream>
#include <cuda_runtime.h>

__global__ void matrixMul(float *A, float *B, float *C, int N) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row < N && col < N) {
        float value = 0;
        for (int k = 0; k < N; k++) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

int main() {
    const int N = 5;
    size_t size = N * N * sizeof(float);
    float *A, *B, *C, *d_A, *d_B, *d_C;

    // Allocate host memory
    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    // Initialize 5x5 matrices with some sample values
    for (int i = 0; i < N * N; i++) {
        A[i] = (i % N) + 1;        // e.g., 1, 2, 3, 4, 5, 1, 2, ...
        B[i] = ((i % N) + 1) * 2;  // e.g., 2, 4, 6, 8, 10, 2, 4, ...
    }

    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy input matrices to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Define thread hierarchy
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize(); // Ensure kernel finishes

    // Copy result back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Display matrices and result
    std::cout << "Matrix A:\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            std::cout << A[i * N + j] << "\t";
        std::cout << "\n";
    }

    std::cout << "\nMatrix B:\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            std::cout << B[i * N + j] << "\t";
        std::cout << "\n";
    }

    std::cout << "\nResult Matrix C (A x B):\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            std::cout << C[i * N + j] << "\t";
        std::cout << "\n";
    }

    // Clean up
    free(A); free(B); free(C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
