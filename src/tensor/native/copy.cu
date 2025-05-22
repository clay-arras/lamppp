#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <cstring>
#include "lamppp/common/assert.hpp"
#include "lamppp/tensor/device_type.hpp"
#include "lamppp/tensor/dispatch_stub.hpp"
#include "lamppp/tensor/dispatch_type.hpp"
#include "lamppp/tensor/native/copy.cuh"
#include "lamppp/tensor/supported_types.hpp"

namespace lmp::tensor::detail::native {

void copy_cpu(DeviceType to_device, const void* src, void* dest, size_t size,
              DataType src_dtype, DataType dest_dtype) {
  switch (to_device) {
    case DeviceType::CPU: {
      LMP_INTERNAL_ASSERT(false, "Not implemented");
      memcpy(dest, src, size);
      break;
    }
    case DeviceType::CUDA: {
      LMP_DISPATCH_ALL_TYPES(src_dtype, [&] {
        using src_type = scalar_t;
        LMP_DISPATCH_ALL_TYPES(dest_dtype, [&] {
          using dest_type = scalar_t;

          void* tmp = nullptr;
          LMP_CUDA_ASSERT(cudaMalloc(&tmp, size * sizeof(src_type)),
                          "copy_cpu to CUDA: cudaMalloc for tmp failed.");
          LMP_CUDA_ASSERT(cudaMemcpy(tmp, src, size * sizeof(src_type),
                                     cudaMemcpyHostToDevice),
                          "copy_cpu to CUDA: cudaMemcpy HtoD for tmp failed.");

          vecCopy<src_type, dest_type>(size, static_cast<const src_type*>(tmp),
                                       static_cast<dest_type*>(dest));

          LMP_CUDA_ASSERT(cudaGetLastError(),
                          "copy_cpu to CUDA: vecCopy kernel failed.");
          LMP_CUDA_ASSERT(cudaFree(tmp),
                          "copy_cpu to CUDA: cudaFree for tmp failed.");
        });
      });
      break;
    }
  }
}

void copy_cuda(DeviceType to_device, const void* src, void* dest, size_t size,
               DataType src_dtype, DataType dest_dtype) {
  switch (to_device) {
    case DeviceType::CPU: {
      LMP_DISPATCH_ALL_TYPES(src_dtype, [&] {
        using src_type = scalar_t;
        LMP_DISPATCH_ALL_TYPES(dest_dtype, [&] {
          using dest_type = scalar_t;

          void* tmp = nullptr;
          LMP_CUDA_ASSERT(cudaMalloc(&tmp, size * sizeof(dest_type)),
                          "copy_cuda to CPU: cudaMalloc for tmp failed.");

          vecCopy<src_type, dest_type>(size, static_cast<const src_type*>(src),
                                       static_cast<dest_type*>(tmp));
          LMP_CUDA_ASSERT(
              cudaGetLastError(),
              "copy_cuda to CPU: vecCopy kernel launch or execution failed.");
          LMP_CUDA_ASSERT(cudaMemcpy(dest, tmp, size * sizeof(dest_type),
                                     cudaMemcpyDeviceToHost),
                          "copy_cuda to CPU: cudaMemcpy DtoH failed.");
          LMP_CUDA_ASSERT(cudaFree(tmp),
                          "copy_cuda to CPU: cudaFree for tmp failed.");
        });
      });
      break;
    }
    case DeviceType::CUDA: {
      LMP_DISPATCH_ALL_TYPES(src_dtype, [&] {
        using src_type = scalar_t;
        LMP_DISPATCH_ALL_TYPES(dest_dtype, [&] {
          using dest_type = scalar_t;

          void* tmp = nullptr;
          LMP_CUDA_ASSERT(cudaMalloc(&tmp, size * sizeof(dest_type)),
                          "copy_cuda to CUDA: cudaMalloc for tmp failed.");

          vecCopy<src_type, dest_type>(size, static_cast<const src_type*>(src),
                                       static_cast<dest_type*>(tmp));

          LMP_CUDA_ASSERT(cudaGetLastError(),
                          "copy_cuda to CUDA: vecCopy kernel failed.");
          LMP_CUDA_ASSERT(cudaMemcpy(dest, tmp, size * sizeof(dest_type),
                                     cudaMemcpyDeviceToDevice),
                          "copy_cuda to CUDA: cudaMemcpy DtoD failed.");
          LMP_CUDA_ASSERT(cudaFree(tmp),
                          "copy_cuda to CUDA: cudaFree for tmp failed.");
        });
      });
      break;
    }
  }
}

template <typename U, typename V>
__global__ void vecCopyKernel(size_t size, const U* in, V* out) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    out[i] = static_cast<V>(in[i]);
  }
}

// TODO: need to make it more clear WHEN something is size and when something is byteSize; should be in function signature
template <typename U, typename V>
void vecCopy(size_t size, const U* in, V* out) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  vecCopyKernel<U, V><<<blocks, threads>>>(size, in, out);
}

// clang-format off

#define INSTANTIATE_COPY(r, product) \
  template void vecCopy<BOOST_PP_SEQ_ELEM(0, product),      /* U */ \
                        BOOST_PP_SEQ_ELEM(1, product)       /* V */ \
                        >(size_t, const BOOST_PP_SEQ_ELEM(0, product)*, \
                                  BOOST_PP_SEQ_ELEM(1, product)*);


#include "lamppp/tensor/supported_types.hpp"
#define TYPES_LIST LMP_TYPES()
BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE_COPY, (TYPES_LIST)(TYPES_LIST))

#undef INSTANTIATE_COPY

// clang-format on

LMP_DEFINE_DISPATCH(copy_stub);

}  // namespace lmp::tensor::detail::native