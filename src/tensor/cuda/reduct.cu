#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cuda/std/array>
#include "lamppp/tensor/cuda/list_ptr.cuh"
#include "lamppp/tensor/cuda/reduct.cuh"

namespace lmp::tensor::detail::cuda {

template <typename PtrList, typename OpFn>
__global__ void vectorized_reduct_kernel(PtrList ptr_, OpFn fn_, size_t size,
                                         size_t axis, const size_t* shape,
                                         const stride_t* strides) {
  size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < size) {
    stride_t outer = strides[axis];
    stride_t inner = strides[axis - 1];
    stride_t idx = (i / outer) * inner + (i % outer);

    auto incr = OpFn::identity;
    for (size_t j = 0; j < shape[axis]; ++j) {
      incr = fn_(incr,
                 ::cuda::std::get<1>(ptr_.fns)(ptr_.data[1], idx + j * outer));
    }
    ptr_.set_Out(i, incr);
  }
}

template <typename PtrList, typename OpFn>
void reduct_kernel_launcher(PtrList ptr_, OpFn fn_, size_t size, size_t axis,
                            const size_t* shape, const stride_t* strides,
                            size_t ndims) {
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;
  ListDevicePtr<stride_t> d_strides(strides, ndims);
  ListDevicePtr<size_t> d_shape(shape, ndims);
  vectorized_reduct_kernel<<<blocks, threads>>>(ptr_, fn_, size, axis,
                                                d_shape.get(), d_strides.get());
  LMP_CUDA_ASSERT(cudaDeviceSynchronize(),
                  "reduct_kernel_launcher: kernel failed.");
}

}  // namespace lmp::tensor::detail::cuda