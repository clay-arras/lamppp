
#include "lamppp/autograd/functions/unary_ops.hpp"
#include <cmath>
#include "lamppp/autograd/functions/overloads.hpp"
#include "lamppp/autograd/functions/unary_decl.hpp"
#include "lamppp/autograd/variable.hpp"
#include "lamppp/common/assert.hpp"
#include "lamppp/common/macros.hpp"
#include "lamppp/tensor/fill_like.hpp"
#include "lamppp/tensor/tensor.hpp"

namespace lmp::autograd::ops {

LMP_FOR_EACH_CARTESIAN_PRODUCT(
    LMP_AUTOGRAD_FN_UNARY_DECL,
    ((NegationBackward, -grad.grad()),
     (ExponentialBackward, grad.data() * grad.grad()),
     (LogarithmBackward, (1 / self.data()) * grad.grad()),
     (SquareRootBackward, (1 / (2 * grad.data())) * grad.grad()),
     (AbsoluteValueBackward,
      ((self.data() > 0.0) - (self.data() < 0.0)) * grad.grad()),
     (SineBackward, grad.grad() * tensor::ops::cos(self.data())),
     (CosineBackward, -1 * tensor::ops::sin(self.data()) * grad.grad()),
     (TangentBackward,
      (1.0 / (tensor::ops::cos(self.data()) * tensor::ops::cos(self.data()))) *
          grad.grad()),
     (ClampBackward,
      (tensor::ones_like(self.data()) * (self.data() > min_val_)) *
          (self.data() < max_val_) * grad.grad())));

LMP_FOR_EACH_CARTESIAN_PRODUCT(
    LMP_AUTOGRAD_FFN_UNARY_DECL,
    ((Negation, tensor::ops::neg), (Exponential, tensor::ops::exp),
     (Logarithm, tensor::ops::log), (SquareRoot, tensor::ops::sqrt),
     (AbsoluteValue, tensor::ops::abs), (Sine, tensor::ops::sin),
     (Cosine, tensor::ops::cos), (Tangent, tensor::ops::tan),
     (Clamp, tensor::ops::clamp, min_val_, max_val_)));

}  // namespace lmp::autograd::ops