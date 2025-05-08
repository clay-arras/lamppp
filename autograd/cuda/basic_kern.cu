#include "basic_kern.cuh"

namespace autograd {

inline namespace cuda {

template <typename U, typename V, typename OutType>
__global__ void vecAddKernel(size_t size, const U* A, const V* B, OutType* C) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    C[i] = A[i] + B[i];
  }
}

template <typename U, typename V, typename OutType>
__global__ void vecSubKernel(size_t size, const U* A, const V* B, OutType* C) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    C[i] = A[i] - B[i];
  }
}

template <typename U, typename V, typename OutType>
__global__ void vecMulKernel(size_t size, const U* A, const V* B, OutType* C) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    C[i] = A[i] * B[i];
  }
}

template <typename U, typename V, typename OutType>
__global__ void vecDivKernel(size_t size, const U* A, const V* B, OutType* C) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    C[i] = A[i] / B[i];
  }
}

template <typename U, typename V, typename OutType>
void vecAdd(size_t size, const U* A, const V* B, OutType* C) {
  size_t bytes = size * sizeof(OutType);
  cudaMalloc(&C, bytes);

  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecAddKernel<U, V, OutType><<<blocks, threads>>>(size, A, B, C);
}

template <typename U, typename V, typename OutType>
void vecSub(size_t size, const U* A, const V* B, OutType* C) {
  size_t bytes = size * sizeof(OutType);
  cudaMalloc(&C, bytes);

  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecSubKernel<U, V, OutType><<<blocks, threads>>>(size, A, B, C);
}

template <typename U, typename V, typename OutType>
void vecMul(size_t size, const U* A, const V* B, OutType* C) {
  size_t bytes = size * sizeof(OutType);
  cudaMalloc(&C, bytes);

  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecMulKernel<U, V, OutType><<<blocks, threads>>>(size, A, B, C);
}

template <typename U, typename V, typename OutType>
void vecDiv(size_t size, const U* A, const V* B, OutType* C) {
  size_t bytes = size * sizeof(OutType);
  cudaMalloc(&C, bytes);

  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecDivKernel<U, V, OutType><<<blocks, threads>>>(size, A, B, C);
}

// #define X(TYPE)                                                            \
//   template void vecAdd<TYPE, TYPE, TYPE>(size_t, const TYPE*, const TYPE*, \
//                                          TYPE*);                           \
//   template void vecSub<TYPE, TYPE, TYPE>(size_t, const TYPE*, const TYPE*, \
//                                          TYPE*);                           \
//   template void vecMul<TYPE, TYPE, TYPE>(size_t, const TYPE*, const TYPE*, \
//                                          TYPE*);                           \
//   template void vecDiv<TYPE, TYPE, TYPE>(size_t, const TYPE*, const TYPE*, \
//                                          TYPE*);
// #include "autograd/engine/supported_types.def"
// #undef X

// clang-format off

// NOTE: this is the most scuffed code I've ever wrote. 
#define TYPES(X, ...) \
    X(__VA_ARGS__, int) \
    X(__VA_ARGS__, float) \
    X(__VA_ARGS__, double)

#define CAST(U, V) \
    template void vecAdd<U, V, U>(size_t, const U*, const V*, U*); \
    template void vecSub<U, V, U>(size_t, const U*, const V*, U*); \
    template void vecMul<U, V, U>(size_t, const U*, const V*, U*); \
    template void vecDiv<U, V, U>(size_t, const U*, const V*, U*); \
    template void vecAdd<V, U, V>(size_t, const V*, const U*, V*); \
    template void vecSub<V, U, V>(size_t, const V*, const U*, V*); \
    template void vecMul<V, U, V>(size_t, const V*, const U*, V*); \
    template void vecDiv<V, U, V>(size_t, const V*, const U*, V*);

#define EXPAND(X) X
#define TYPES1() TYPES
#define TYPES2(...) TYPES1 EXPAND(())(__VA_ARGS__)

EXPAND(TYPES(TYPES2, CAST))
// 


}  // namespace cuda
}  // namespace autograd