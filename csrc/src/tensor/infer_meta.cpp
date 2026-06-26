#include "lamppp/tensor/infer_meta.hpp"
#include "lamppp/common/assert.hpp"
#include "lamppp/tensor/data_type.hpp"
#include "lamppp/tensor/tensor_impl.hpp"
#include "lamppp/tensor/utils/align_utils.hpp"

namespace lmp::tensor::detail {

OpMeta infer_unary(const TensorImpl* a) {
  OpMeta m;
  m.dtype = a->type();
  m.size = a->numel();
  m.shape = a->shape();
  m.expand = false;
  return m;
}

OpMeta infer_binary(const TensorImpl* a, const TensorImpl* b) {
  LMP_INTERNAL_ASSERT(a->device() == b->device())
      << "Should have asserted already";
  OpMeta m;
  m.dtype = type_upcast(a->type(), b->type());
  m.size = a->numel();
  m.shape = a->shape();
  m.expand = (a->shape() != b->shape());
  if (m.expand) {
    detail::AlignUtil expand_dims(a->shape(), b->shape());
    m.size = expand_dims.aligned_size_;
    m.shape = expand_dims.aligned_shape_;
  }
  return m;
}

OpMeta infer_reduct(const TensorImpl* a, size_t axis) {
  OpMeta m;
  m.dtype = a->type();
  m.size = a->numel();
  m.shape = a->shape();
  m.expand = false;
  m.size /= m.shape[axis];
  m.shape[axis] = 1;
  return m;
}

}  // namespace lmp::tensor::detail
