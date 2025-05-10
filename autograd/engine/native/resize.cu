#include "autograd/engine/native/resize.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <new>

namespace autograd {

DEFINE_DISPATCH(resize_stub);

void resize_cpu(DataPtr dptr, size_t old_byte_size, size_t new_byte_size) {
  void* ptr = ::operator new(new_byte_size);
  std::memcpy(ptr, dptr.data, std::min(old_byte_size, new_byte_size));

  (*(dptr.deallocator))(dptr.data);
  dptr.data = ptr;
}

void resize_cuda(DataPtr dptr, size_t old_byte_size, size_t new_byte_size) {
  void* ptr = nullptr;
  cudaMalloc(&ptr, new_byte_size);
  cudaMemcpy(ptr, dptr.data, std::min(old_byte_size, new_byte_size),
             cudaMemcpyDeviceToDevice);
  (*(dptr.deallocator))(dptr.data);
  dptr.data = ptr;
}

}  // namespace autograd
