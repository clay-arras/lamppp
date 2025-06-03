#include "lamppp/autograd/functions/expand_ops.hpp"
#include <memory>
#include "lamppp/autograd/function.hpp"
#include "lamppp/autograd/functions/binary_decl.hpp"
#include "lamppp/autograd/grad_utils.hpp"
#include "lamppp/autograd/variable.hpp"
#include "lamppp/common/assert.hpp"
#include "lamppp/common/macros.hpp"
#include "lamppp/tensor/fill_like.hpp"

namespace lmp::autograd::ops {

LMP_FOR_EACH_CARTESIAN_PRODUCT(
    LMP_AUTOGRAD_FN_BINARY_DECL,
    ((AddBackward, detail::sum_broadcast_axis(grad.grad(), self.data().shape()),
      detail::sum_broadcast_axis(grad.grad(), other.data().shape())),
     (SubtractBackward,
      detail::sum_broadcast_axis(grad.grad(), self.data().shape()),
      detail::sum_broadcast_axis(-grad.grad(), other.data().shape())),
     (MultiplyBackward,
      detail::sum_broadcast_axis(other.data() * grad.grad(),
                                 self.data().shape()),
      detail::sum_broadcast_axis(self.data() * grad.grad(),
                                 other.data().shape())),
     (DivideBackward,
      detail::sum_broadcast_axis(grad.grad() / other.data(),
                                 self.data().shape()),
      detail::sum_broadcast_axis((-1.0) *
                                     (grad.data() * grad.grad() / other.data()),
                                 other.data().shape())),
     (PowerBackward,
      detail::sum_broadcast_axis(grad.grad() * other.data() *
                                     tensor::ops::pow(self.data(),
                                                      other.data() - 1),
                                 self.data().shape()),
      detail::sum_broadcast_axis(grad.grad() * grad.data() *
                                     tensor::ops::log(self.data()),
                                 other.data().shape())),
     (EqualBackward,
      ((LMP_INTERNAL_ASSERT(false) << "Not implemented"), tensor::Tensor()),
      ((LMP_INTERNAL_ASSERT(false) << "Not implemented"), tensor::Tensor())),
     (LessBackward,
      ((LMP_INTERNAL_ASSERT(false) << "Not implemented"), tensor::Tensor()),
      ((LMP_INTERNAL_ASSERT(false) << "Not implemented"), tensor::Tensor())),
     (LessEqualBackward,
      ((LMP_INTERNAL_ASSERT(false) << "Not implemented"), tensor::Tensor()),
      ((LMP_INTERNAL_ASSERT(false) << "Not implemented"), tensor::Tensor())),
     (NotEqualBackward,
      ((LMP_INTERNAL_ASSERT(false) << "Not implemented"), tensor::Tensor()),
      ((LMP_INTERNAL_ASSERT(false) << "Not implemented"), tensor::Tensor())),
     (GreaterBackward,
      ((LMP_INTERNAL_ASSERT(false) << "Not implemented"), tensor::Tensor()),
      ((LMP_INTERNAL_ASSERT(false) << "Not implemented"), tensor::Tensor())),
     (GreaterEqualBackward,
      ((LMP_INTERNAL_ASSERT(false) << "Not implemented"), tensor::Tensor()),
      ((LMP_INTERNAL_ASSERT(false) << "Not implemented"), tensor::Tensor()))));

LMP_FOR_EACH_CARTESIAN_PRODUCT(
    LMP_AUTOGRAD_FFN_BINARY_DECL,
    ((Add, tensor::ops::add), (Subtract, tensor::ops::sub),
     (Multiply, tensor::ops::mul), (Divide, tensor::ops::div),
     (Power, tensor::ops::pow), (Equal, tensor::ops::eq),
     (Less, tensor::ops::lt), (LessEqual, tensor::ops::le),
     (NotEqual, tensor::ops::ne), (Greater, tensor::ops::gt),
     (GreaterEqual, tensor::ops::ge)));

}  // namespace lmp::autograd::ops