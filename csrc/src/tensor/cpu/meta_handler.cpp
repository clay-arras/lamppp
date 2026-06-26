#include "lamppp/tensor/cpu/meta_handler.hpp"
#include "lamppp/common/assert.hpp"
#include "lamppp/tensor/device_type.hpp"
#include "lamppp/tensor/infer_meta.hpp"

namespace lmp::tensor::detail {

template <>
UnaryMetaHandler::TensorMetaHandler(const TensorImpl* a) : inTens_({a}) {
  OpMeta m = infer_unary(a);
  outDtype_ = m.dtype;
  outSize_ = m.size;
  outShape_ = m.shape;
  expand_ = m.expand;
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
    : inTens_({a, b}) {
  OpMeta m = infer_binary(a, b);
  outDtype_ = m.dtype;
  outSize_ = m.size;
  outShape_ = m.shape;
  expand_ = m.expand;

  LMP_DISPATCH_ALL_TYPES(outDtype_, [&] {
    using out_dtype_t = scalar_t;
    LMP_DISPATCH_ALL_TYPES(a->type(), [&] {
      using arg1_dtype_t = scalar_t;
      LMP_DISPATCH_ALL_TYPES(b->type(), [&] {
        using arg2_dtype_t = scalar_t;
        Storage out_st(outSize_ * sizeof(out_dtype_t), a->device());
        outTen_ = std::make_unique<TensorImpl>(out_st, outShape_, outDtype_);
        if (expand_) {  // TODO(astronaut): if I comment this out, the code is 2-4 times faster. idk why
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
    : inTens_({a}) {
  OpMeta m = infer_reduct(a, axis);
  outDtype_ = m.dtype;
  outSize_ = m.size;
  outShape_ = m.shape;
  expand_ = m.expand;
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
