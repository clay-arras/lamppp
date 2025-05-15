#include "include/lamppp/tensor/native/empty.cuh"

namespace lmp::tensor::detail::native {

LMP_DEFINE_DISPATCH(empty_stub);

DataPtr empty_cpu(size_t byte_size) {
  void* ptr_ = ::operator new(byte_size);
  return DataPtr(ptr_, [](void* ptr) { ::operator delete(ptr); });
}
DataPtr empty_cuda(size_t byte_size) {
  void* ptr_ = nullptr;
  cudaError_t err = cudaMalloc(&ptr_, byte_size);
  assert(err == cudaSuccess && "empty_cuda: cudaMalloc failed.");
  return DataPtr(ptr_, [](void* ptr) { cudaFree(ptr); });
}

}  // namespace lmp::tensor::detail::native