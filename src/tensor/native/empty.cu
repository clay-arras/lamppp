#include "lamppp/tensor/native/empty.cuh"

namespace lmp::tensor::detail::native {

LMP_DEFINE_DISPATCH(empty_fn, empty_stub);

DataPtr empty_cpu(size_t byte_size) {
  void* ptr_ = static_cast<void*>(new char[byte_size]);
  return DataPtr(ptr_, [](void* ptr) { delete[] static_cast<char*>(ptr); });
}
DataPtr empty_cuda(size_t byte_size) {
  void* ptr_ = nullptr;
  LMP_CUDA_ASSERT(cudaMalloc(&ptr_, byte_size),
                  "empty_cuda: cudaMalloc failed.");
  return DataPtr(ptr_, [](void* ptr) { cudaFree(ptr); });
}

LMP_REGISTER_DISPATCH(empty_stub, DeviceType::CPU, empty_cpu);
LMP_REGISTER_DISPATCH(empty_stub, DeviceType::CUDA, empty_cuda);

}  // namespace lmp::tensor::detail::native