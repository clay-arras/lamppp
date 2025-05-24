#include "lamppp/tensor/cpu/reduct.hpp"
#include <array>
#include "lamppp/tensor/align_utils.hpp"

namespace lmp::tensor::detail::cpu {

template <typename PtrList, typename OpFn>
void vectorized_reduct_kernel(PtrList ptr_, OpFn fn_, size_t i, size_t axis,
                              const size_t* shape, const stride_t* strides) {
  stride_t outer = strides[axis];
  stride_t inner = strides[axis - 1];
  stride_t idx = (i / outer) * inner + (i % outer);

  auto incr = OpFn::identity;
  for (size_t j = 0; j < shape[axis]; ++j) {
    incr = fn_(incr, ::std::get<1>(ptr_.fns)(ptr_.data[1], idx + j * outer));
  }
  ptr_.set_Out(i, incr);
}

template <typename PtrList, typename OpFn>
void reduct_kernel_launcher(PtrList ptr_, OpFn fn_, size_t size, size_t axis,
                            const size_t* shape, const stride_t* strides,
                            size_t ndims) {
#pragma omp parallel for simd
  for (size_t i = 0; i < size; i++) {
    vectorized_reduct_kernel(ptr_, fn_, i, axis, shape, strides);
  }
}

}  // namespace lmp::tensor::detail::cpu