#include "include/lamppp/tensor/native/resize.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <new>

namespace lmp::tensor::detail::native {

LMP_DEFINE_DISPATCH(resize_stub);

void resize_cpu(DataPtr dptr, size_t old_byte_size, size_t new_byte_size) {
  void* ptr = ::operator new(new_byte_size);
  std::memcpy(ptr, dptr.data(), std::min(old_byte_size, new_byte_size));

  auto deleter = std::get_deleter<std::function<void(void*)>>(dptr.ptr);
  dptr = DataPtr(ptr, *deleter);
}

void resize_cuda(DataPtr dptr, size_t old_byte_size, size_t new_byte_size) {
  void* ptr = nullptr;
  cudaMalloc(&ptr, new_byte_size);
  cudaMemcpy(ptr, dptr.data(), std::min(old_byte_size, new_byte_size),
             cudaMemcpyDeviceToDevice);

  auto deleter = std::get_deleter<std::function<void(void*)>>(dptr.ptr);
  dptr = DataPtr(ptr, *deleter);
}

}  // namespace lmp::tensor::detail::native
