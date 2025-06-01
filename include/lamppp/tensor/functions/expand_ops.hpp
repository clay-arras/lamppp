#pragma once

#include "lamppp/tensor/device_type.hpp"
#include "lamppp/tensor/dispatch_stub.hpp"
#include "lamppp/tensor/tensor.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::ops {

using add_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using sub_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using mul_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using div_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using pow_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using eq_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using ne_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using ge_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using le_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using gt_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using lt_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);

LMP_DECLARE_DISPATCH(add_fn, add_stub);
LMP_DECLARE_DISPATCH(sub_fn, sub_stub);
LMP_DECLARE_DISPATCH(mul_fn, mul_stub);
LMP_DECLARE_DISPATCH(div_fn, div_stub);
LMP_DECLARE_DISPATCH(pow_fn, pow_stub);
LMP_DECLARE_DISPATCH(eq_fn, eq_stub);
LMP_DECLARE_DISPATCH(ne_fn, ne_stub);
LMP_DECLARE_DISPATCH(ge_fn, ge_stub);
LMP_DECLARE_DISPATCH(le_fn, le_stub);
LMP_DECLARE_DISPATCH(gt_fn, gt_stub);
LMP_DECLARE_DISPATCH(lt_fn, lt_stub);

Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);
Tensor pow(const Tensor& a, const Tensor& b);
Tensor eq(const Tensor& a, const Tensor& b);
Tensor ne(const Tensor& a, const Tensor& b);
Tensor ge(const Tensor& a, const Tensor& b);
Tensor le(const Tensor& a, const Tensor& b);
Tensor gt(const Tensor& a, const Tensor& b);
Tensor lt(const Tensor& a, const Tensor& b);

}  // namespace lmp::tensor::ops