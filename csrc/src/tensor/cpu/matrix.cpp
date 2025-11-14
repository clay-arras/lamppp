#include "lamppp/tensor/cpu/matrix.hpp"
#include "lamppp/common/macros.hpp"
#include "lamppp/tensor/data_type.hpp"

namespace lmp::tensor::detail::cpu {

template <typename U, typename V, typename OutType>
void cpuMatmulKernel(const U* A, const V* B, OutType* C, size_t m, size_t n,
                     size_t k, size_t i, size_t j) {
  OutType sum = 0;
#pragma omp parallel for schedule(static) reduction(+ : sum)
  for (size_t t = 0; t < k; t++) {
    sum += static_cast<OutType>(A[(i * k) + t]) *
           static_cast<OutType>(B[(n * t) + j]);
  }
  C[(i * n) + j] = sum;
}

template <typename T>
void cpuTransposeKernel(const T* in, T* out, size_t m, size_t n, size_t i,
                        size_t j) {
  out[(j * m) + i] = in[(i * n) + j];
}

template <typename U, typename V, typename OutType>
void cpuMatMul(const U* A, const V* B, OutType* C, size_t m, size_t n,
               size_t k) {
#pragma omp parallel for collapse(2) schedule(static)
  for (size_t i = 0; i < m; i++)
    for (size_t j = 0; j < n; j++)
      cpuMatmulKernel<U, V, OutType>(A, B, C, m, n, k, i, j);
}

template <typename T>
void cpuTranspose(const T* in, T* out, size_t m, size_t n) {
#pragma omp parallel for collapse(2) schedule(static)
  for (size_t i = 0; i < m; i++)
    for (size_t j = 0; j < n; j++)
      cpuTransposeKernel<T>(in, out, m, n, i, j);
}

#define INSTANTIATE_MATMUL(arg1_type, arg2_type, out_type) \
  template void cpuMatMul<arg1_type, arg2_type, out_type>( \
      const arg1_type*, const arg2_type*, out_type*, size_t, size_t, size_t);
#define INSTANTIATE_TRANSPOSE(type) \
  template void cpuTranspose<type>(const type*, type*, size_t, size_t);

LMP_FOR_EACH_CARTESIAN_PRODUCT(INSTANTIATE_MATMUL, LMP_LIST_TYPES,
                               LMP_LIST_TYPES, LMP_LIST_TYPES);
LMP_FOR_EACH_CARTESIAN_PRODUCT(INSTANTIATE_TRANSPOSE, LMP_LIST_TYPES);

#undef INSTANTIATE_MATMUL
#undef INSTANTIATE_TRANSPOSE

}  // namespace lmp::tensor::detail::cpu