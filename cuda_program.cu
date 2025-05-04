#include <iostream>
#include <cuda.h>
using namespace std;

// =============================
// Vector Addition Kernel
// =============================
__global__ void vectorAdd(const float* A, const float* B, float* C, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
        C[idx] = A[idx] + B[idx];
}

// =============================
// Matrix Multiplication Kernel
// =============================
__global__ void matrixMul(const float* A, const float* B, float* C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    if (row < width && col < width) {
        for (int k = 0; k < width; ++k)
            sum += A[row * width + k] * B[k * width + col];
        C[row * width + col] = sum;
    }
}

int main() {
    int choice;
    cout << "CUDA Operations:\n";
    cout << "1. Vector Addition\n2. Matrix Multiplication\nChoose (1 or 2): ";
    cin >> choice;

    if (choice == 1) {
        // VECTOR ADDITION
        int n;
        cout << "Enter number of elements: ";
        cin >> n;

        size_t size = n * sizeof(float);
        float *h_A = new float[n], *h_B = new float[n], *h_C = new float[n];

        cout << "Enter elements of vector A:\n";
        for (int i = 0; i < n; i++) cin >> h_A[i];

        cout << "Enter elements of vector B:\n";
        for (int i = 0; i < n; i++) cin >> h_B[i];

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, size);
        cudaMalloc(&d_B, size);
        cudaMalloc(&d_C, size);

        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;
        vectorAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C, n);

        cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

        cout << "\nVector Addition Result (first 10 values):\n";
        for (int i = 0; i < min(n, 10); i++)
            cout << h_A[i] << " + " << h_B[i] << " = " << h_C[i] << endl;

        delete[] h_A; delete[] h_B; delete[] h_C;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    } else if (choice == 2) {
        // MATRIX MULTIPLICATION
        int N;
        cout << "Enter size of square matrix (NxN): ";
        cin >> N;

        size_t size = N * N * sizeof(float);
        float *h_A = new float[N * N], *h_B = new float[N * N], *h_C = new float[N * N];

        cout << "Enter elements of matrix A (" << N*N << " elements row-wise):\n";
        for (int i = 0; i < N * N; i++) cin >> h_A[i];

        cout << "Enter elements of matrix B (" << N*N << " elements row-wise):\n";
        for (int i = 0; i < N * N; i++) cin >> h_B[i];

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, size);
        cudaMalloc(&d_B, size);
        cudaMalloc(&d_C, size);

        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((N + 15) / 16, (N + 15) / 16);
        matrixMul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

        cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

        cout << "\nMatrix Multiplication Result (first 10 values row-wise):\n";
        for (int i = 0; i < min(N*N, 10); i++)
            cout << h_C[i] << " ";
        cout << endl;

        delete[] h_A; delete[] h_B; delete[] h_C;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    } else {
        cout << "Invalid choice.\n";
    }

    return 0;
}

/*

nvcc cuda_vector_matrix.cu -o cuda_program

./cuda_program


*/