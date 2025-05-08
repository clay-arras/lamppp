#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include "autograd/engine/device_type.hpp"

struct Allocator {
  virtual DeviceType device() = 0;
  virtual void* allocate(size_t bytes) = 0;
  virtual std::shared_ptr<std::function<void(void*)>> deallocate() = 0;
};
