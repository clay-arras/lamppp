#include <cstring>
#include "autograd/engine/device_type.hpp"
#include "autograd/engine/dispatch_stub.hpp"
#include "copy.cuh"

namespace autograd {

DEFINE_DISPATCH(copy_stub);

void copy_cpu(DeviceType to_device, const void* src, void* dest, size_t size,
              DataType src_dtype, DataType dest_dtype) {
  switch (to_device) {
    case DeviceType::CPU: {
      memcpy(dest, src, size);
      break;
    }
    case DeviceType::CUDA: {
      cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
      break;
    }
  }
}

void copy_cuda(DeviceType to_device, const void* src, void* dest, size_t size,
               DataType src_dtype, DataType dest_dtype) {
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