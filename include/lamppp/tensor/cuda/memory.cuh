#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cuda/std/array>
#include "lamppp/tensor/data_ptr.hpp"
#include "lamppp/tensor/data_type.hpp"
#include "lamppp/tensor/device_type.hpp"
#include "lamppp/tensor/scalar.hpp"

namespace lmp::tensor::detail::cuda {

/// @internal
void copy_cuda(DeviceType to_device, const void* src, void* dest, size_t size,
               DataType src_dtype, DataType dest_dtype);
DataPtr empty_cuda(size_t byte_size);
void fill_cuda(void* ptr, size_t size, Scalar t, DataType type);
void resize_cuda(DataPtr dptr, size_t old_byte_size, size_t new_byte_size);

template <typename U, typename V>
__global__ void cudaVecCopyKernel(size_t size, const U* in, V* out);
template <typename U, typename V>
void cudaVecCopy(size_t size, const U* in, V* out);

void vecCopyHostToDevice(const void* src, void* dest, size_t size,
              DataType src_dtype, DataType dest_dtype);
/// @endinternal

}  // namespace lmp::tensor::detail::cuda