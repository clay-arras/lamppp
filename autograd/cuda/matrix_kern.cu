#include "matrix_kern.cuh"

namespace autograd {

inline namespace cuda {

// Kernels moved outside anonymous namespace
template <typename T>
__global__ void cudaMatmulKernel(const T* A, const T* B, T* C, int m, int n, int k) {
    int i = threadIdx.x + (blockIdx.x * blockDim.x);
    int j = threadIdx.y + (blockIdx.y * blockDim.y);

    if (i < m && j < n) {
        T sum = 0;
        for (int t=0; t<k; t++) { // NOTE: A is MxK, B is KxN, C is MxN
            // sum += A[(i*k) + t] * B[(n*t) + j]; // row major implementation
            sum += A[i + m*t] * B[t + k*j]; // TODO(nlin): this can be made faster but whatever
        }
        C[(j*m) + i] = sum;
    }
}

template <typename T>
__global__ void cudaTransposeKernel(const T* in,
                                    T* out,
                                    int m,
                                    int n) {
    int i = threadIdx.x + (blockIdx.x * blockDim.x);
    int j = threadIdx.y + (blockIdx.y * blockDim.y);

    if (i < m && j < n) {
        out[(i*n) + j] = in[(j*m) + i];
    }
}

template <typename T>
void cudaMatMul(const T* A, const T* B, T* C, int m, int n, int k) {
  T *d_a;
  T *d_b;
  T *d_c;
  size_t bytes_a = m * k * sizeof(T);
  size_t bytes_b = k * n * sizeof(T);
  size_t bytes_c = m * n * sizeof(T);

  cudaMalloc(&d_a, bytes_a);
  cudaMalloc(&d_b, bytes_b);
  cudaMalloc(&d_c, bytes_c);
  cudaMemcpy(d_a, A, bytes_a, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, bytes_b, cudaMemcpyHostToDevice);

  dim3 threads(16, 16);
  dim3 blocks((m + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y);
  cudaMatmulKernel<<<blocks, threads>>>(d_a, d_b, d_c, m, n, k);

  cudaMemcpy(C, d_c, bytes_c, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

template <typename T>
void cudaTranspose(const T* in,
                              T* out,
                              int m,
                              int n) {
  T *d_in;
  T *d_out;
  size_t bytes_in = m * n * sizeof(T);
  size_t bytes_out = m * n * sizeof(T);

  cudaMalloc(&d_in, bytes_in);
  cudaMalloc(&d_out, bytes_out);
  cudaMemcpy(d_in, in, bytes_in, cudaMemcpyHostToDevice);

  dim3 threads(16, 16);
  dim3 blocks((m + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y);
  cudaTransposeKernel<<<blocks, threads>>>(d_in, d_out, m, n);

  cudaMemcpy(out, d_out, bytes_out, cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
}

#define X(TYPE) template void cudaMatMul<TYPE>(const TYPE*, const TYPE*, TYPE*, int, int, int); \
                 template void cudaTranspose<TYPE>(const TYPE*, TYPE*, int, int);
#include "autograd/engine/supported_types.def"
#undef  X

} // namespace cuda

} // namespace autograd