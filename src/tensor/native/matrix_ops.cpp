#include "lamppp/tensor/native/matrix_ops.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::ops {

LMP_DEFINE_DISPATCH(matmul_fn, matmul_stub);
LMP_DEFINE_DISPATCH(transpose_fn, transpose_stub);

Tensor matmul(const Tensor& a, const Tensor& b) {
  LMP_CHECK(a.device() == b.device()) << "Tensors must be on the same device";
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      matmul_stub()(a.device(), *detail::UnsafeTensorAccessor::getImpl(a),
                    *detail::UnsafeTensorAccessor::getImpl(b))));
}

Tensor transpose(const Tensor& a) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      transpose_stub()(a.device(), *detail::UnsafeTensorAccessor::getImpl(a))));
}

}  // namespace lmp::tensor::ops
