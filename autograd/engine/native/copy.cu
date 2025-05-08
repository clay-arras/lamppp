#include <cstring>
#include "autograd/engine/device_type.hpp"
#include "autograd/engine/dispatch_stub.hpp"
#include "copy.cuh"

namespace autograd {

DEFINE_DISPATCH(copy_stub);

void copy_cpu(const void* src, void* dest, size_t size, DeviceType to_device) {
  switch (to_device) {
    case DeviceType::CPU: {
      memcpy(dest, src, size);
      break;
    }
    case DeviceType::CUDA: {
      cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
      break;
    }
  }
}

void copy_cuda(const void* src, void* dest, size_t size, DeviceType to_device) {
  switch (to_device) {
    case DeviceType::CPU: {
      cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
      break;
    }
    case DeviceType::CUDA: {
      cudaMemcpy(dest, src, size, cudaMemcpyDeviceToDevice);
      break;
    }
  }
}

}  // namespace autograd