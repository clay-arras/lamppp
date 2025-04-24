
namespace autograd {

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

}

}