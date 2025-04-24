namespace autograd {

namespace cuda {

namespace {

__global__ void add(int size, const float* A, const float* B, float* C) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(int size, const float* A, const float* B, float* C) {
  float *d_a, *d_b, *d_out;
  size_t bytes = n * sizeof(float);

  cudaMalloc(&d_a,    bytes);
  cudaMalloc(&d_b,    bytes);
  cudaMalloc(&d_out,  bytes);

  cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks  = (n + threads - 1) / threads;
  vectorAddKernel<<<blocks, threads>>>(d_a, d_b, d_out, n);

  cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
}

__global__ void sub(int size, const float* A, const float* B, float* C) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = A[i] - B[i];
    }
}

__global__ void mul(int size, const float* A, const float* B, float* C) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = A[i] * B[i];
    }
}

__global__ void div(int size, const float* A, const float* B, float* C) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = A[i] / B[i];
    }
}

}

}

}