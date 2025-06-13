#include "lamppp/tensor/cpu/meta_handler.hpp"
#include "lamppp/common/assert.hpp"
#include "lamppp/tensor/device_type.hpp"

namespace lmp::tensor::detail {

template <>
UnaryMetaHandler::TensorMetaHandler(const TensorImpl* a)
    : inTens_({a}),
      outDtype_(a->type()),
      outSize_(a->numel()),
      outShape_(a->shape()),
      expand_(false) {
  LMP_DISPATCH_ALL_TYPES(outDtype_, [&] {
    using out_dtype_t = scalar_t;
    LMP_DISPATCH_ALL_TYPES(a->type(), [&] {
      using arg_dtype_t = scalar_t;
      Storage out_st(outSize_ * sizeof(out_dtype_t), a->device());
      outTen_ = std::make_unique<TensorImpl>(out_st, outShape_, outDtype_);
    });
  });
}

template <>
BinaryMetaHandler::TensorMetaHandler(const TensorImpl* a, const TensorImpl* b)
    : inTens_({a, b}),
      outDtype_(type_upcast(a->type(), b->type())),
      outSize_(a->numel()),
      outShape_(a->shape()),
      expand_(false) {
  LMP_INTERNAL_ASSERT(a->device() == b->device())
      << "Should have asserted already";
  expand_ = (a->shape() != b->shape());
  if (expand_) {
    detail::AlignUtil expand_dims(a->shape(), b->shape());
    outSize_ = expand_dims.aligned_size_;
    outShape_ = expand_dims.aligned_shape_;
  }

  LMP_DISPATCH_ALL_TYPES(outDtype_, [&] {
    using out_dtype_t = scalar_t;
    LMP_DISPATCH_ALL_TYPES(a->type(), [&] {
      using arg1_dtype_t = scalar_t;
      LMP_DISPATCH_ALL_TYPES(b->type(), [&] {
        using arg2_dtype_t = scalar_t;
        Storage out_st(outSize_ * sizeof(out_dtype_t), a->device());
        outTen_ = std::make_unique<TensorImpl>(out_st, outShape_, outDtype_);
        if (expand_) {  // TODO: if I comment this out, the code is 2-4 times faster. idk why
          outOffset_ = offset_util_stub_2()(
              a->device(),
              ::std::array<const TensorImpl*, BinaryMetaHandler::kNumElem>{a,
                                                                           b},
              *outTen_.get());
        }
      });
    });
  });
}

template <>
ReductMetaHandler::TensorMetaHandler(const TensorImpl* a, size_t axis)
    : inTens_({a}),
      outDtype_(a->type()),
      outSize_(a->numel()),
      outShape_(a->shape()),
      expand_(false) {
  outSize_ /= outShape_[axis];
  outShape_[axis] = 1;
  LMP_DISPATCH_ALL_TYPES(outDtype_, [&] {
    using out_dtype_t = scalar_t;
    LMP_DISPATCH_ALL_TYPES(a->type(), [&] {
      using arg_dtype_t = scalar_t;
      Storage out_st(outSize_ * sizeof(out_dtype_t), a->device());
      outTen_ = std::make_unique<TensorImpl>(out_st, outShape_, outDtype_);
    });
  });
}

}  // namespace lmp::tensor::detail