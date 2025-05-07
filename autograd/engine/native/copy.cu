#include <cstring>
#include "autograd/engine/data_ptr.hpp"
#include "autograd/engine/device_type.hpp"
#include "autograd/engine/dispatch_stub.hpp"
#include "copy.cuh"
#include "empty.cuh"

namespace autograd {

DEFINE_DISPATCH(copy_stub);

DataPtr copy_cpu(DataPtr src, size_t size, DeviceType to_device) {
  DataPtr out = empty_stub(to_device, size);

  switch (to_device) {
    case DeviceType::CPU: {
      memcpy(out.data, src.data, size);
      return out;
    }
    case DeviceType::CUDA: {
      cudaMemcpy(out.data, src.data, size, cudaMemcpyDeviceToHost);
      return out;
    }
  }
}

DataPtr copy_cuda(DataPtr src, size_t size, DeviceType to_device) {
  DataPtr out = empty_stub(to_device, size);

  switch (to_device) {
    case DeviceType::CPU: {
      cudaMemcpy(out.data, src.data, size, cudaMemcpyDeviceToHost);
      return out;
    }
    case DeviceType::CUDA: {
      cudaMemcpy(out.data, src.data, size, cudaMemcpyDeviceToDevice);
      return out;
    }
  }
}

}  // namespace autograd