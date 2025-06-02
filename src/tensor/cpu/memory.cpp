#include "lamppp/tensor/cpu/memory.hpp"
#include "lamppp/tensor/native/memory_ops.hpp"
#include <cstring>
#include "lamppp/tensor/align_utils.hpp"
#include "lamppp/tensor/dispatch_type.hpp"

namespace lmp::tensor::detail::cpu {

void fill_cpu(void* ptr, size_t size, Scalar t, DataType type) {
  LMP_DISPATCH_ALL_TYPES(type, [&]() {
    scalar_t* data = static_cast<scalar_t*>(ptr);
    std::fill(data, data + size, static_cast<scalar_t>(t));
  });
}
void resize_cpu(DataPtr dptr, size_t old_byte_size, size_t new_byte_size) {
  void* ptr = ::operator new(new_byte_size);
  std::memcpy(ptr, dptr.data(), std::min(old_byte_size, new_byte_size));

  auto deleter = std::get_deleter<std::function<void(void*)>>(dptr.ptr);
  dptr = DataPtr(ptr, *deleter);
}
DataPtr empty_cpu(size_t byte_size) {
  void* ptr_ = static_cast<void*>(new char[byte_size]);
  return DataPtr(ptr_, [](void* ptr) { delete[] static_cast<char*>(ptr); });
}

LMP_REGISTER_DISPATCH(ops::empty_stub, DeviceType::CPU, empty_cpu);
LMP_REGISTER_DISPATCH(ops::resize_stub, DeviceType::CPU, resize_cpu);
LMP_REGISTER_DISPATCH(ops::fill_stub, DeviceType::CPU, fill_cpu);

}  // namespace lmp::tensor::detail::cpu