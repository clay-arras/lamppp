#pragma once

#include "include/lamppp/tensor/functions/basic_ops.hpp"
#include "include/lamppp/tensor/functions/binary_ops.hpp"
#include "include/lamppp/tensor/tensor.hpp"

namespace autograd {

struct TensorOpFact {
  template <Tensor (*OpTag)(const Tensor&, const Tensor&)>
  static inline Tensor binary_tensor_op(const Tensor& a, const Tensor& b) {
    return (*OpTag)(a, b);
  }

  template <Tensor (*OpTag)(const Tensor&, const Tensor&)>
  static inline Tensor binary_tensor_op(const Tensor& tensor, Scalar scalar) {
    Tensor scalar_tensor(std::vector<Scalar>(tensor.size(), scalar),
                         tensor.shape(), tensor.device(), tensor.type());
    return binary_tensor_op<OpTag>(tensor, scalar_tensor);
  }

  template <Tensor (*OpTag)(const Tensor&, const Tensor&)>
  static inline Tensor binary_tensor_op(Scalar scalar, const Tensor& tensor) {
    Tensor scalar_tensor(std::vector<Scalar>(tensor.size(), scalar),
                         tensor.shape(), tensor.device(), tensor.type());
    return binary_tensor_op<OpTag>(scalar_tensor, tensor);
  }
};

#define DECL_BINARY_OP(op, tag)                                    \
  inline Tensor operator op(const Tensor& a, const Tensor& b) {    \
    return TensorOpFact::binary_tensor_op<&tag>(a, b);             \
  }                                                                \
  inline Tensor operator op(const Tensor& tensor, Scalar scalar) { \
    return TensorOpFact::binary_tensor_op<&tag>(tensor, scalar);   \
  }                                                                \
  inline Tensor operator op(Scalar scalar, const Tensor& tensor) { \
    return TensorOpFact::binary_tensor_op<&tag>(scalar, tensor);   \
  }

#define FORALL_BINARY_OPS(_) \
  _(+, add)                  \
  _(-, sub)                  \
  _(*, mul)                  \
  _(/, div)                  \
  _(==, equal)               \
  _(!=, not_equal)           \
  _(>=, greater_equal)       \
  _(<=, less_equal)          \
  _(>, greater)              \
  _(<, less)

FORALL_BINARY_OPS(DECL_BINARY_OP)

#undef FORALL_BINARY_OPS
#undef DECL_BINARY_OP

}  // namespace autograd