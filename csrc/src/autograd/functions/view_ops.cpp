#include "lamppp/autograd/functions/view_ops.hpp"
#include <cmath>
#include "lamppp/autograd/functions/unary_decl.hpp"
#include "lamppp/autograd/variable.hpp"
#include "lamppp/common/macros.hpp"
#include "lamppp/tensor/native/shape_ops.hpp"
#include "lamppp/tensor/tensor.hpp"
#include "lamppp/tensor/utils/fill_like.hpp"

namespace lmp::autograd::ops {

LMP_FOR_EACH_CARTESIAN_PRODUCT(
    LMP_AUTOGRAD_FN_UNARY_DECL,
    ((ToBackward, grad.grad().to(self.data().device())),
     (ReshapeBackward, grad.grad().reshape(self.data().shape())),
     (SqueezeBackward, grad.grad().expand_dims(axis)),
     (ExpandDimsBackward, grad.grad().squeeze(axis)), ));

LMP_FOR_EACH_CARTESIAN_PRODUCT(
    LMP_AUTOGRAD_FFN_UNARY_DECL,
    ((To, tensor::ops::to, device), (Reshape, tensor::ops::reshape, shape),
     (Squeeze, tensor::ops::squeeze, axis),
     (ExpandDims, tensor::ops::expand_dims, axis), ));

}  // namespace lmp::autograd::ops