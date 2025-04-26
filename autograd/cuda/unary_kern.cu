#include "unary_kern.cuh"
#include <cmath>

namespace autograd {

inline namespace cuda {

namespace {

__global__ void vecExpKernel(int size, float* in, float* out) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        out[i] = expf(in[i]);
    }
}

__global__ void vecLogKernel(int size, float* in, float* out) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        out[i] = logf(in[i]);
    }
}

__global__ void vecReluKernel(int size, float* in, float* out) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        out[i] = fmaxf(0.0F, in[i]);
    }
}

} // anonymous namespace

extern "C" void vecExp(int size, const float* in, float* out) {
  float *d_in;
  float *d_out;
  size_t bytes = size * sizeof(float);

  cudaMalloc(&d_in, bytes);
  cudaMalloc(&d_out, bytes);
  cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  vecExpKernel<<<blocks, threads>>>(size, d_in, d_out);

  cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
}

extern "C" void vecLog(int size, const float* in, float* out) {
  float *d_in;
  float *d_out;
  size_t bytes = size * sizeof(float);

  cudaMalloc(&d_in, bytes);
  cudaMalloc(&d_out, bytes);
  cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  vecLogKernel<<<blocks, threads>>>(size, d_in, d_out);

  cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
}

extern "C" void vecRelu(int size, const float* in, float* out) {
  float *d_in;
  float *d_out;
  size_t bytes = size * sizeof(float);

  cudaMalloc(&d_in, bytes);
  cudaMalloc(&d_out, bytes);
  cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  vecReluKernel<<<blocks, threads>>>(size, d_in, d_out);

  cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
}

} // namespace cuda

} // namespace autograd