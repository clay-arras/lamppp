#include "lamppp/tensor/lazy/functions/elementwise_binary.hpp"
#include "lamppp/tensor/infer_meta.hpp"
#include "lamppp/tensor/native/expand_ops.hpp"
#include "lamppp/tensor/storage.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor {

std::shared_ptr<TensorImpl> ElementwiseBinaryFn::infer_output() const {
  detail::OpMeta m = detail::infer_binary(inputs[0].get(), inputs[1].get());
  Storage empty(0, inputs[0]->device());                  // 0-byte, on-device
  return std::make_shared<TensorImpl>(empty, m.shape, m.dtype);
}

bool ElementwiseBinaryFn::is_fusible() const {
  return inputs[0]->shape() == inputs[1]->shape();  // same-shape => fusible; broadcast => boundary
}

void AddFn::run_eager(TensorImpl& out) {
  TensorImpl res = ops::add_stub()(inputs[0]->device(), *inputs[0], *inputs[1]);
  out.set_realized(res.storage());
}

void SubFn::run_eager(TensorImpl& out) {
  TensorImpl res = ops::sub_stub()(inputs[0]->device(), *inputs[0], *inputs[1]);
  out.set_realized(res.storage());
}

void MulFn::run_eager(TensorImpl& out) {
  TensorImpl res = ops::mul_stub()(inputs[0]->device(), *inputs[0], *inputs[1]);
  out.set_realized(res.storage());
}

void DivFn::run_eager(TensorImpl& out) {
  TensorImpl res = ops::div_stub()(inputs[0]->device(), *inputs[0], *inputs[1]);
  out.set_realized(res.storage());
}

void PowFn::run_eager(TensorImpl& out) {
  TensorImpl res = ops::pow_stub()(inputs[0]->device(), *inputs[0], *inputs[1]);
  out.set_realized(res.storage());
}

void EqFn::run_eager(TensorImpl& out) {
  TensorImpl res = ops::eq_stub()(inputs[0]->device(), *inputs[0], *inputs[1]);
  out.set_realized(res.storage());
}

void NeFn::run_eager(TensorImpl& out) {
  TensorImpl res = ops::ne_stub()(inputs[0]->device(), *inputs[0], *inputs[1]);
  out.set_realized(res.storage());
}

void GeFn::run_eager(TensorImpl& out) {
  TensorImpl res = ops::ge_stub()(inputs[0]->device(), *inputs[0], *inputs[1]);
  out.set_realized(res.storage());
}

void LeFn::run_eager(TensorImpl& out) {
  TensorImpl res = ops::le_stub()(inputs[0]->device(), *inputs[0], *inputs[1]);
  out.set_realized(res.storage());
}

void GtFn::run_eager(TensorImpl& out) {
  TensorImpl res = ops::gt_stub()(inputs[0]->device(), *inputs[0], *inputs[1]);
  out.set_realized(res.storage());
}

void LtFn::run_eager(TensorImpl& out) {
  TensorImpl res = ops::lt_stub()(inputs[0]->device(), *inputs[0], *inputs[1]);
  out.set_realized(res.storage());
}

}  // namespace lmp::tensor
