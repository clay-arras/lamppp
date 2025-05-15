#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <cstring>
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
      assert(false && "Not implemented");
      memcpy(dest, src, size);
      break;
    }
    case DeviceType::CUDA: {
      LMP_DISPATCH_ALL_TYPES(src_dtype, [&] {
        using src_type = scalar_t;
        LMP_DISPATCH_ALL_TYPES(dest_dtype, [&] {
          using dest_type = scalar_t;

          void* tmp = nullptr;
          cudaError_t err = cudaMalloc(&tmp, size * sizeof(src_type));
          assert(err == cudaSuccess &&
                 "copy_cpu to CUDA: cudaMalloc for tmp failed.");

          err = cudaMemcpy(tmp, src, size * sizeof(src_type),
                           cudaMemcpyHostToDevice);
          assert(err == cudaSuccess &&
                 "copy_cpu to CUDA: cudaMemcpy HtoD for tmp failed.");

          vecCopy<src_type, dest_type>(size, static_cast<const src_type*>(tmp),
                                       static_cast<dest_type*>(dest));

          cudaError_t post_kernel_err = cudaGetLastError();
          assert(
              post_kernel_err == cudaSuccess &&
              "copy_cpu to CUDA: vecCopy kernel launch or execution failed.");

          err = cudaFree(tmp);
          assert(err == cudaSuccess &&
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
          cudaError_t err = cudaMalloc(&tmp, size * sizeof(dest_type));
          assert(err == cudaSuccess &&
                 "copy_cuda to CPU: cudaMalloc for tmp failed.");

          vecCopy<src_type, dest_type>(size, static_cast<const src_type*>(src),
                                       static_cast<dest_type*>(tmp));
          err = cudaGetLastError();
          assert(
              err == cudaSuccess &&
              "copy_cuda to CPU: vecCopy kernel launch or execution failed.");

          err = cudaMemcpy(dest, tmp, size * sizeof(dest_type),
                           cudaMemcpyDeviceToHost);
          assert(err == cudaSuccess &&
                 "copy_cuda to CPU: cudaMemcpy DtoH failed.");

          err = cudaFree(tmp);
          assert(err == cudaSuccess &&
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
          cudaError_t err = cudaMalloc(&tmp, size * sizeof(dest_type));
          assert(err == cudaSuccess &&
                 "copy_cuda to CUDA (via tmp): cudaMalloc for tmp failed.");

          vecCopy<src_type, dest_type>(size, static_cast<const src_type*>(src),
                                       static_cast<dest_type*>(tmp));
          err = cudaGetLastError();
          assert(err == cudaSuccess &&
                 "copy_cuda to CUDA (via tmp): vecCopy kernel launch or "
                 "execution failed.");

          err = cudaMemcpy(dest, tmp, size * sizeof(dest_type),
                           cudaMemcpyDeviceToDevice);
          assert(err == cudaSuccess &&
                 "copy_cuda to CUDA (via tmp): cudaMemcpy DtoD failed.");

          err = cudaFree(tmp);
          assert(err == cudaSuccess &&
                 "copy_cuda to CUDA (via tmp): cudaFree for tmp failed.");
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