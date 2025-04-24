#include "unary_kern.cuh"
#include <cmath>

namespace autograd {

inline namespace cuda {

namespace {

__global__ void exp(int size, float* in, float* out) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        out[i] = std::exp(in[i]);
    }
}

__global__ void log(int size, float* in, float* out) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        out[i] = std::log(in[i]);
    }
}

__global__ void relu(int size, float* in, float* out) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        out[i] = std::max(0.0F, in[i]);
    }
}

} // anonymous namespace

extern "C" void vecExp(int size, float* in, float* out) {
  float *d_in;
  float *d_out;
  size_t bytes = size * sizeof(float);

  cudaMalloc(&d_in, bytes);
  cudaMalloc(&d_out, bytes);
  cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  exp<<<blocks, threads>>>(size, d_in, d_out);

  cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
}

extern "C" void vecLog(int size, float* in, float* out) {
  float *d_in;
  float *d_out;
  size_t bytes = size * sizeof(float);

  cudaMalloc(&d_in, bytes);
  cudaMalloc(&d_out, bytes);
  cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  log<<<blocks, threads>>>(size, d_in, d_out);

  cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
}

extern "C" void vecRelu(int size, float* in, float* out) {
  float *d_in;
  float *d_out;
  size_t bytes = size * sizeof(float);

  cudaMalloc(&d_in, bytes);
  cudaMalloc(&d_out, bytes);
  cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  relu<<<blocks, threads>>>(size, d_in, d_out);

  cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
}

} // namespace cuda

} // namespace autograd