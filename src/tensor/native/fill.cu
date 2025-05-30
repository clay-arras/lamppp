#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include "lamppp/tensor/data_type.hpp"
#include "lamppp/tensor/dispatch_type.hpp"
#include "lamppp/tensor/scalar.hpp"
#include "lamppp/tensor/native/fill.cuh"

namespace lmp::tensor::detail::native {

LMP_DEFINE_DISPATCH(fill_fn, fill_stub);

void fill_cpu(void* ptr, size_t size, Scalar t, DataType type) {
  LMP_DISPATCH_ALL_TYPES(type, [&]() {
    scalar_t* data = static_cast<scalar_t*>(ptr);
    std::fill(data, data + size, static_cast<scalar_t>(t));
  });
}

void fill_cuda(void* ptr, size_t size, Scalar t, DataType type) {
  LMP_DISPATCH_ALL_TYPES(type, [&]() {
    thrust::device_ptr<scalar_t> data(static_cast<scalar_t*>(ptr));
    thrust::fill(data, data + size, static_cast<scalar_t>(t));
    LMP_CUDA_ASSERT(cudaGetLastError(), "fill_cuda: thrust::fill failed.");
  });
}

LMP_REGISTER_DISPATCH(fill_stub, DeviceType::CPU, fill_cpu);
LMP_REGISTER_DISPATCH(fill_stub, DeviceType::CUDA, fill_cuda);

}  // namespace lmp::tensor::detail::native