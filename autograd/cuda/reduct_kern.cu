#include "reduct_kern.cuh"

namespace autograd {

inline namespace cuda {

template <typename T>
__global__ void vecSumKernel(const T* in,
                             T* out,
                             const int* shape,
                             int* stride,
                             int axis, 
                             int outSize) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < outSize) {
        size_t outer = stride[axis];
        size_t inner = stride[axis+1];            
        size_t idx   = (i / outer) * inner
                     + (i % outer);

        T sum = 0;                      
        for (int j = 0; j < shape[axis]; ++j) {
            sum += in[idx + j * outer];           
        }

        out[i] = sum;
    }
}

template <typename T>
__global__ void vecMaxKernel(const T* in,
                             T* out,
                             const int* shape,
                             int* stride,
                             int axis, 
                             int outSize) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < outSize) {
        size_t outer = stride[axis];
        size_t inner = stride[axis+1];            
        size_t idx   = (i / outer) * inner
                     + (i % outer);

        T max = 0.0f;                      
        for (int j = 0; j < shape[axis]; ++j) {
            max = fmaxf(max, in[idx + j * outer]); 
        }

        out[i] = max;
    }
}

template <typename T>
void vecSum(const T* in,
                       T* out,
                       const int* shape,
                       int axis,
                       int ndims) {
    int totalSize = 1;
    for (int i = 0; i < ndims; ++i) {
        totalSize *= shape[i];
    }
    int outSize = totalSize / shape[axis];
    int* h_stride = new int[ndims + 1];

    h_stride[0] = 1;
    for (int i = 1; i <= ndims; i++) {
        h_stride[i] = h_stride[i - 1] * shape[i-1];
    }
    
    T *d_in, *d_out;
    int *d_shape, *d_stride;
    
    cudaMalloc(&d_in, totalSize * sizeof(T));
    cudaMalloc(&d_out, outSize * sizeof(T));
    cudaMalloc(&d_shape, ndims * sizeof(int));
    cudaMalloc(&d_stride, (ndims + 1) * sizeof(int));
    
    cudaMemcpy(d_in, in, totalSize * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shape, shape, ndims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stride, h_stride, (ndims + 1) * sizeof(int), cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (outSize + threads - 1) / threads;
    vecSumKernel<<<blocks, threads>>>(d_in, d_out, d_shape, d_stride, axis, outSize);
    
    cudaMemcpy(out, d_out, outSize * sizeof(T), cudaMemcpyDeviceToHost);
    
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_shape);
    cudaFree(d_stride);
    
    delete[] h_stride;
}

template <typename T>
void vecMax(const T* in,
                       T* out,
                       const int* shape,
                       int axis, 
                       int ndims) {
    int totalSize = 1;
    for (int i = 0; i < ndims; ++i) {
        totalSize *= shape[i];
    }
    int outSize = totalSize / shape[axis];
    int* h_stride = new int[ndims + 1];

    h_stride[0] = 1;
    for (int i = 1; i <= ndims; i++) {
        h_stride[i] = h_stride[i - 1] * shape[i-1];
    }
    
    T *d_in, *d_out;
    int *d_shape, *d_stride;
    
    cudaMalloc(&d_in, totalSize * sizeof(T));
    cudaMalloc(&d_out, outSize * sizeof(T));
    cudaMalloc(&d_shape, ndims * sizeof(int));
    cudaMalloc(&d_stride, (ndims + 1) * sizeof(int));
    
    cudaMemcpy(d_in, in, totalSize * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shape, shape, ndims * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stride, h_stride, (ndims + 1) * sizeof(int), cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (outSize + threads - 1) / threads;
    vecSumKernel<<<blocks, threads>>>(d_in, d_out, d_shape, d_stride, axis, outSize);
    
    cudaMemcpy(out, d_out, outSize * sizeof(T), cudaMemcpyDeviceToHost);
    
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_shape);
    cudaFree(d_stride);
    
    delete[] h_stride;

}

template void vecSum<float>(const float* in, float* out, const int* shape, int axis, int ndims);
template void vecMax<float>(const float* in, float* out, const int* shape, int axis, int ndims);

} // namespace cuda

} // namespace autograd