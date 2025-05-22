#include "lamppp/tensor/cuda/meta_handler.cuh"
#include "lamppp/tensor/cuda/offset_util.cuh"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::detail::cuda::internal {

TensorMetaHandler::TensorMetaHandler(tensor_list in)
    : inTens(in),
      outDtype_(static_cast<DataType>(0)),
      outSize_(in[0].size()),
      outShape_(in[0].shape()) {
  for (TensorImpl ten : inTens) {
    outDtype_ = type_upcast(outDtype_, ten.type());
  }
}

void TensorMetaHandler::handle_expand_op() {
  LMP_INTERNAL_ASSERT(inTens.size() == 2, "Expected exactly 2 input tensors.");
  detail::AlignUtil expand_dims(inTens[0].shape(), inTens[1].shape());
  outSize_ = expand_dims.aligned_size_;
  outShape_ = expand_dims.aligned_shape_;
  LMP_DISPATCH_ALL_TYPES(outDtype_, [&] {
    using out_dtype_t = scalar_t;
    LMP_DISPATCH_ALL_TYPES(inTens[0].type(), [&] {
      using arg1_dtype_t = scalar_t;
      LMP_DISPATCH_ALL_TYPES(inTens[1].type(), [&] {
        using arg2_dtype_t = scalar_t;
        Storage out_st(outSize_ * sizeof(out_dtype_t), DeviceType::CUDA);
        outTen = std::make_unique<TensorImpl>(out_st, outShape_, outDtype_);

        outOffset = std::make_unique<OffsetUtil<2>>(
            ::std::array<const TensorImpl*, 2>{&inTens[0], &inTens[1]},
            *outTen.get());
      });
    });
  });
}

void TensorMetaHandler::handle_unary_op() {
  LMP_INTERNAL_ASSERT(inTens.size() == 1, "Expected exactly 1 input tensors.");
  LMP_DISPATCH_ALL_TYPES(outDtype_, [&] {
    using out_dtype_t = scalar_t;
    LMP_DISPATCH_ALL_TYPES(inTens[0].type(), [&] {
      using arg_dtype_t = scalar_t;
      Storage out_st(outSize_ * sizeof(out_dtype_t), DeviceType::CUDA);
      outTen = std::make_unique<TensorImpl>(out_st, outShape_, outDtype_);
    });
  });
}

void TensorMetaHandler::handle_reduct_op(size_t axis) {
  LMP_INTERNAL_ASSERT(inTens.size() == 1, "Expected exactly 1 input tensors.");
  outSize_ /= outShape_[axis];
  outShape_[axis] = 1;
  LMP_DISPATCH_ALL_TYPES(outDtype_, [&] {
    using out_dtype_t = scalar_t;
    LMP_DISPATCH_ALL_TYPES(inTens[0].type(), [&] {
      using arg_dtype_t = scalar_t;
      Storage out_st(outSize_ * sizeof(out_dtype_t), DeviceType::CUDA);
      outTen = std::make_unique<TensorImpl>(out_st, outShape_, outDtype_);
    });
  });
}

TensorImpl TensorMetaHandler::out() const noexcept {
  return *outTen.get();
}

tensor_list TensorMetaHandler::in() const noexcept {
  return inTens;
}

const OffsetUtil<2>* TensorMetaHandler::offset() const noexcept {
  return outOffset.get();
}

}  // namespace lmp::tensor::detail::cuda::internal