#include "lamppp/tensor/fill_like.hpp"
#include <algorithm>
#include <random>
#include "lamppp/tensor/dispatch_type.hpp"
#include "lamppp/tensor/scalar.hpp"
#include "lamppp/tensor/tensor.hpp"

namespace lmp::tensor {

Tensor full_like(const Tensor& tensor, Scalar scalar) {
  return LMP_DISPATCH_ALL_TYPES(tensor.type(), [&] {
    Storage storage(tensor.numel() * sizeof(scalar_t), tensor.device());
    TensorImpl impl(storage, tensor.shape(), tensor.type());
    impl.fill(scalar);
    return detail::UnsafeTensorAccessor::fromImpl(
        std::make_shared<TensorImpl>(std::move(impl)));
  });
}

Tensor ones_like(const Tensor& tensor) {
  return full_like(tensor, 1);
}

Tensor zeros_like(const Tensor& tensor) {
  return full_like(tensor, 0);
}

}  // namespace lmp::tensor