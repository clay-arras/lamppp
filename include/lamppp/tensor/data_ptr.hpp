#pragma once

#include <functional>
#include <iostream>
#include <memory>

namespace lmp::tensor::detail {

/// @internal 
/// @todo have it crash in the DEBUG mode, otherwise catch it in production
/**
 * @brief Smart pointer wrapper for tensor data with custom deleter support
 * 
 * @details DataPtr wraps raw pointers with automatic memory management and
 * custom deallocators for different allocation strategies (CPU, CUDA, etc.).
 *
 * @note that DataPtr does not actually define any of the allocation strategies.
 * they are defined in empty_stub()
 *
 * @see empty_stub
 */
struct DataPtr {
  std::shared_ptr<void> ptr;

  DataPtr() = default;
  DataPtr(void* data_ptr, std::function<void(void*)> dealloc)
      : ptr(data_ptr, [dealloc](void* data) {
          try {
            dealloc(data);
          } catch (const std::exception& e) {
            std::cerr << "DataPtr deleter error: " << e.what() << "\n";
          } catch (...) {
            std::cerr << "DataPtr deleter threw non-std::exception\n";
          }
        }) {}

  void* data() const { return ptr.get(); }
};
/// @endinternal

}  // namespace lmp::tensor::detail