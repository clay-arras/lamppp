#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cassert>
#include <cmath>
#include <cuda/std/detail/libcxx/include/array>
#include <memory>
#include <vector>
#include "lamppp/tensor/cuda/list_ptr.cuh"
#include "lamppp/tensor/data_type.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::detail::cuda {

using tensor_list = std::vector<lmp::tensor::TensorImpl>;

using UnaryOpPtrList = ::cuda::std::array<void*, 2>;
using BinaryOpPtrList = ::cuda::std::array<void*, 3>;

template <typename OutType, typename InType>
class LogFunctor {
 public:
  __device__ __host__ void operator()(UnaryOpPtrList ptrs, size_t index) {
    OutType* out_data = static_cast<OutType*>(ptrs[0]);
    const InType* in_data = static_cast<const InType*>(ptrs[1]);
    out_data[index] =
        static_cast<OutType>(::log(static_cast<double>(in_data[index])));
  }
};

namespace internal {

template <size_t N>
::cuda::std::array<void*, N + 1> pack_tens(tensor_list tens, void* out);

class TensorMetaHandler {  // TODO: make it s.t. broadcasting is done here
 public:
  explicit TensorMetaHandler(tensor_list in) : inTens(in) {
    DataType out_dtype =
        static_cast<DataType>(0);  // get the lowest priority DType
    size_t out_size = in[0].size();
    std::vector<size_t> out_shape = in[0].shape();
    for (TensorImpl ten : in) {
      out_dtype = type_upcast(out_dtype, ten.type());
      assert(ten.size() == out_size && "TensorMetaHandler: size mismatch");
      assert(
          ten.shape() == out_shape &&
          "TensorMetaHandler: shape mismatch");  // for now, assuming shapes are all same
    }
    outDtype_ = out_dtype;
    outSize_ = out_size;
    outShape_ = out_shape;
  }

  void handle_binary_op() {
    assert(false && "Not implemented yet");
    assert(inTens.size() == 2);
    LMP_DISPATCH_ALL_TYPES(outDtype_, [&] {
      using out_dtype_t = scalar_t;
      LMP_DISPATCH_ALL_TYPES(inTens[0].type(), [&] {
        using arg1_dtype_t = scalar_t;
        LMP_DISPATCH_ALL_TYPES(inTens[1].type(), [&] {
          using arg2_dtype_t = scalar_t;
          Storage out_st(outSize_ * sizeof(out_dtype_t), DeviceType::CUDA);
          outTen = std::make_unique<TensorImpl>(out_st, outShape_, outDtype_);
        });
      });
    });
  }

  void handle_unary_op() {
    assert(inTens.size() == 1);
    LMP_DISPATCH_ALL_TYPES(outDtype_, [&] {
      using out_dtype_t = scalar_t;
      LMP_DISPATCH_ALL_TYPES(inTens[0].type(), [&] {
        using arg_dtype_t = scalar_t;
        Storage out_st(outSize_ * sizeof(out_dtype_t), DeviceType::CUDA);
        outTen = std::make_unique<TensorImpl>(out_st, outShape_, outDtype_);
      });
    });
  }

  TensorImpl out() const { return *outTen.get(); }
  tensor_list in() const { return inTens; }

 private:
  DataType outDtype_;
  size_t outSize_;
  std::vector<size_t> outShape_;

  std::unique_ptr<TensorImpl> outTen;
  tensor_list inTens;
};

}  // namespace internal

template <typename PtrList, typename OpFn>
__global__ void vectorized_unary_kernel(PtrList ptr_, OpFn& fn_, size_t size);

template <typename PtrList, typename OpFn>
void unary_kernel_launcher(PtrList ptr_, OpFn& fn_, size_t size);

template <template <typename, typename> class OpFunctor>
void unary_gpu_kernel(const internal::TensorMetaHandler& meta);

TensorImpl log_cuda(const TensorImpl& a);

}  // namespace lmp::tensor::detail::cuda