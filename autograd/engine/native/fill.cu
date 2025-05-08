#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include "autograd/engine/data_type.hpp"
#include "autograd/engine/dispatch_type.hpp"
#include "autograd/engine/scalar.hpp"

#include "fill.cuh"

namespace autograd {

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

}  // namespace autograd