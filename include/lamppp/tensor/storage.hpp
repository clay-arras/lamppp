#pragma once

#include <cstdlib>
#include <cstring>
#include <memory>
#include <ostream>
#include "data_ptr.hpp"
#include "lamppp/tensor/device_type.hpp"
#include "lamppp/tensor/native/memory_ops.hpp"

namespace lmp::tensor {

/// @internal
/**
 * @brief  Low-level data manager for Tensor and TensorImpl
 *
 * @details Storage is what manages the lifetime of the data that Tensor points to, in
 * the form of an array of bytes.
 * 
 * Storage maintains three core properties:
 * - A DataPtr containing the raw data pointer and its associated deleter
 * - The allocated size in bytes
 * - The target device type (CPU, CUDA, etc.)
 * 
 * @note Storage is type-agnostic and operates solely on raw bytes (a void*). Type information
 * is maintained at the Tensor/TensorImpl level, in the form of DataType. Data interpretation
 * must be handled using LMP_DISPATCH_ALL_TYPES at higher levels.
 * 
 * @warning Direct manipulation of Storage should be avoided in user code. Use Tensor
 * operations instead for type-safe data access and manipulation.
 *
 * @see Tensor, DataPtr
 */
class Storage {
 public:
  explicit Storage(size_t byte_size, DeviceType device)
      : impl_(std::make_shared<StorageImpl>(byte_size, device)) {}

  void* data() const noexcept;
  size_t byte_size() const noexcept;
  DeviceType device() const noexcept;

  void resize_(size_t nsize);
  friend std::ostream& operator<<(std::ostream& os, const Storage& obj);

 private:
  class StorageImpl;
  std::shared_ptr<StorageImpl> impl_;
};

class Storage::StorageImpl {
 public:
  explicit StorageImpl(size_t byte_size, DeviceType device)
      : data_ptr_(ops::empty_stub()(device, byte_size)),
        byte_size_(byte_size),
        device_(device) {}

  void* data() const noexcept;
  size_t byte_size() const noexcept;
  DeviceType device() const noexcept;

  void resize_(size_t nsize);
  void print_(std::ostream& os);

 private:
  detail::DataPtr data_ptr_;
  size_t byte_size_;
  DeviceType device_;
};
/// @endinternal

}  // namespace lmp::tensor