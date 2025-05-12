#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include "include/lamppp/tensor/data_type.hpp"
#include "include/lamppp/tensor/dispatch_type.hpp"
#include "include/lamppp/tensor/scalar.hpp"

#include "include/lamppp/tensor/native/fill.cuh"

namespace lmp::tensor::detail::native {

DEFINE_DISPATCH(fill_stub);

void fill_cpu(void* ptr, size_t size, Scalar t, DataType type) {
  DISPATCH_ALL_TYPES(type, [&]() {
    scalar_t* data = static_cast<scalar_t*>(ptr);
    std::fill(data, data + size, static_cast<scalar_t>(t));
  });
}

void fill_cuda(void* ptr, size_t size, Scalar t, DataType type) {
  DISPATCH_ALL_TYPES(type, [&]() {
    scalar_t* data = static_cast<scalar_t*>(ptr);
    thrust::fill(data, data + size, static_cast<scalar_t>(t));
  });
}

}  // namespace lmp::tensor::detail::native