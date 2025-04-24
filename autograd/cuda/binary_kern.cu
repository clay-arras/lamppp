namespace autograd {

namespace {

__global__ void equal(int size, const float* A, const float* B, float* C) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = (A[i] == B[i]) ? 1.0F : 0.0F;
    }
}

__global__ void not_equal(int size, const float* A, const float* B, float* C) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = (A[i] != B[i]) ? 1.0F : 0.0F;
    }
}

__global__ void greater_equal(int size, const float* A, const float* B, float* C) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = (A[i] >= B[i]) ? 1.0F : 0.0F;
    }
}

__global__ void less_equal(int size, const float* A, const float* B, float* C) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = (A[i] <= B[i]) ? 1.0F : 0.0F;
    }
}

__global__ void greater_than(int size, const float* A, const float* B, float* C) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = (A[i] > B[i]) ? 1.0F : 0.0F;
    }
}

__global__ void less_than(int size, const float* A, const float* B, float* C) {
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < size) {
        C[i] = (A[i] < B[i]) ? 1.0F : 0.0F;
    }
}


} // namespace

} // namespace autograd