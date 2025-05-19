#include "lamppp/tensor/cuda/meta_util.cuh"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::detail::cuda::internal {

TensorMetaHandler::TensorMetaHandler(tensor_list in)
    : inTens(in),
      outDtype_(static_cast<DataType>(0)),
      outSize_(in[0].size()),
      outShape_(in[0].shape()) {
  for (TensorImpl ten : in) {
    outDtype_ = type_upcast(outDtype_, ten.type());
    assert(ten.size() == outSize_ && "TensorMetaHandler: size mismatch");
    assert(ten.shape() == outShape_ && "TensorMetaHandler: shape mismatch");
  }
}

void TensorMetaHandler::handle_binary_op() {
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

void TensorMetaHandler::handle_unary_op() {
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

TensorImpl TensorMetaHandler::out() const {
  return *outTen.get();
}

tensor_list TensorMetaHandler::in() const {
  return inTens;
}

}  // namespace lmp::tensor::detail::cuda::internal