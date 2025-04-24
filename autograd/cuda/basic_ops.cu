namespace autograd {

namespace {

__global__ void add(int size, const float* A, const float* B, float* C) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = A[i] + B[i];
    }
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