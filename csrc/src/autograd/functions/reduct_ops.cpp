#include "lamp3/autograd/functions/reduct_ops.hpp"
#include "lamp3/autograd/functions/unary_decl.hpp"
#include "lamp3/autograd/variable.hpp"
#include "lamp3/common/assert.hpp"
#include "lamp3/common/macros.hpp"
#include "lamp3/tensor/utils/fill_like.hpp"

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
