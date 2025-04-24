#include "basic_kern.cuh"

namespace autograd {

inline namespace cuda {

namespace {

__global__ void vecAddKernel(int size, const float* A, const float* B, float* C) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = A[i] + B[i];
    }
}

__global__ void vecSubKernel(int size, const float* A, const float* B, float* C) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = A[i] - B[i];
    }
}

__global__ void vecMulKernel(int size, const float* A, const float* B, float* C) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = A[i] * B[i];
    }
}

__global__ void vecDivKernel(int size, const float* A, const float* B, float* C) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = A[i] / B[i];
    }
}

} // anonymous namespace

extern "C" void vecAdd(int size, const float* A, const float* B, float* C) {
  float *d_a;
  float *d_b;
  float *d_c;
  size_t bytes = size * sizeof(float);

  cudaMalloc(&d_a,    bytes);
  cudaMalloc(&d_b,    bytes);
  cudaMalloc(&d_c,  bytes);
  cudaMemcpy(d_a, A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, bytes, cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks  = (size + threads - 1) / threads;
  vecAddKernel<<<blocks, threads>>>(size, d_a, d_b, d_c);

  cudaMemcpy(C, d_c, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

extern "C" void vecSub(int size, const float* A, const float* B, float* C) {
  float *d_a;
  float *d_b;
  float *d_c;
  size_t bytes = size * sizeof(float);

  cudaMalloc(&d_a,    bytes);
  cudaMalloc(&d_b,    bytes);
  cudaMalloc(&d_c,  bytes);
  cudaMemcpy(d_a, A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, bytes, cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks  = (size + threads - 1) / threads;
  vecSubKernel<<<blocks, threads>>>(size, d_a, d_b, d_c);

  cudaMemcpy(C, d_c, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

extern "C" void vecMul(int size, const float* A, const float* B, float* C) {
  float *d_a;
  float *d_b;
  float *d_c;
  size_t bytes = size * sizeof(float);

  cudaMalloc(&d_a,    bytes);
  cudaMalloc(&d_b,    bytes);
  cudaMalloc(&d_c,  bytes);
  cudaMemcpy(d_a, A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, bytes, cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks  = (size + threads - 1) / threads;
  vecMulKernel<<<blocks, threads>>>(size, d_a, d_b, d_c);

  cudaMemcpy(C, d_c, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

extern "C" void vecDiv(int size, const float* A, const float* B, float* C) {
  float *d_a;
  float *d_b;
  float *d_c;
  size_t bytes = size * sizeof(float);

  cudaMalloc(&d_a,    bytes);
  cudaMalloc(&d_b,    bytes);
  cudaMalloc(&d_c,  bytes);
  cudaMemcpy(d_a, A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, bytes, cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks  = (size + threads - 1) / threads;
  vecDivKernel<<<blocks, threads>>>(size, d_a, d_b, d_c);

  cudaMemcpy(C, d_c, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

}

}