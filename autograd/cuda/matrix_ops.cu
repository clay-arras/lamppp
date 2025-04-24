namespace autograd {

namespace {

__global__ void matmul(const float* A, const float* B, float* C, int m, int n, int k) {
    int i = threadIdx.x + (blockIdx.x * blockDim.x);
    int j = threadIdx.y + (blockIdx.y * blockDim.y);

    if (i < m && j < n) {
        C[(i*n) + j] = 0;
        for (int t=0; t<k; t++) { // NOTE: A is MxK, B is KxN, C is MxN
            C[(i*n) + j] += A[(i*k) + t] * B[(n*t) + j]; // TODO(nlin): this can be made faster but whatever
        }
    }
}

}

}