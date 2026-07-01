#include "lamp3/autograd/functions/view_ops.hpp"

#include <cmath>

#include "lamp3/autograd/functions/unary_decl.hpp"
#include "lamp3/autograd/variable.hpp"
#include "lamp3/common/macros.hpp"
#include "lamp3/tensor/native/shape_ops.hpp"
#include "lamp3/tensor/tensor.hpp"
#include "lamp3/tensor/utils/fill_like.hpp"

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