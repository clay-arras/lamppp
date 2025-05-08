#pragma once

#ifndef CUDA_ALLOCATOR_H
#define CUDA_ALLOCATOR_H

#include <memory>
#include "autograd/engine/device_type.hpp"
#include <cuda_runtime.h>
#include "autograd/engine/allocator.hpp"
#include <cstddef>
#include <functional>

#ifdef __cplusplus

namespace autograd {

struct CudaAllocator : Allocator {
  DeviceType device() override {
    return DeviceType::CUDA;
  }
  void* allocate(size_t bytes) override {
    void *ptr;
    cudaMalloc(&ptr, bytes);
    return ptr;
  }
  std::shared_ptr<std::function<void(void*)>> deallocate() override {
    return std::make_shared<std::function<void(void*)>>([](void* ptr) {
      cudaFree(ptr);
    });
  }
};

}

#endif

#endif  // CUDA_ALLOCATOR_H