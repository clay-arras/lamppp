#include "lamppp/autograd/functions/reduct_ops.hpp"
#include "lamppp/autograd/functions/unary_decl.hpp"
#include "lamppp/autograd/variable.hpp"
#include "lamppp/common/assert.hpp"
#include "lamppp/common/macros.hpp"
#include "lamppp/tensor/fill_like.hpp"

namespace lmp::autograd::ops {

LMP_FOR_EACH_CARTESIAN_PRODUCT(
    LMP_AUTOGRAD_FN_UNARY_DECL,
    ((SummationBackward, tensor::ones_like(self.data()) * grad.grad()),
     (MaximumBackward, grad.grad() * tensor::ops::eq(self.data(), grad.data())),
     (MinimumBackward, grad.grad() * tensor::ops::eq(self.data(), grad.data())),
     (ProductBackward, grad.grad() * (grad.data() / self.data())), ));

LMP_FOR_EACH_CARTESIAN_PRODUCT(LMP_AUTOGRAD_FFN_UNARY_DECL,
                               ((Summation, tensor::ops::sum, axis_),
                                (Maximum, tensor::ops::max, axis_),
                                (Minimum, tensor::ops::min, axis_),
                                (Product, tensor::ops::prod, axis_), ));

}  // namespace lmp::autograd::ops
