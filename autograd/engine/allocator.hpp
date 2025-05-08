#pragma once

#include <memory>
#include "autograd/engine/device_type.hpp"
#ifndef ALLOCATOR_H
#define ALLOCATOR_H

#include <cstddef>
#include <functional>

struct Allocator {
  virtual DeviceType device() = 0;
  virtual void* allocate(size_t bytes) = 0;
  virtual std::shared_ptr<std::function<void(void*)>> deallocate() = 0;
};

#endif  // ALLOCATOR_H