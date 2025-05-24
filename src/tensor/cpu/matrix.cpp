#include "lamppp/tensor/cpu/matrix.hpp"
#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <cstdint>

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

#define INSTANTIATE_MATMUL(r, product)                                      \
  template void                                                             \
  cpuMatMul<BOOST_PP_SEQ_ELEM(0, product), BOOST_PP_SEQ_ELEM(1, product),   \
            BOOST_PP_SEQ_ELEM(2, product)>(                                 \
      const BOOST_PP_SEQ_ELEM(0, product)*,                                 \
      const BOOST_PP_SEQ_ELEM(1, product)*, BOOST_PP_SEQ_ELEM(2, product)*, \
      size_t, size_t, size_t);

#define INSTANTIATE_TRANSPOSE(r, data, elem) \
  template void cpuTranspose<elem>(const elem*, elem*, size_t, size_t);

#include "lamppp/tensor/supported_types.hpp"
#define TYPES_LIST LMP_TYPES()
BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE_MATMUL,
                              (TYPES_LIST)(TYPES_LIST)(TYPES_LIST))
BOOST_PP_SEQ_FOR_EACH(INSTANTIATE_TRANSPOSE, , TYPES_LIST)

#undef INSTANTIATE_MATMUL
#undef INSTANTIATE_TRANSPOSE

}  // namespace lmp::tensor::detail::cpu