#pragma once

#include "lamppp/tensor/device_type.hpp"
#include "lamppp/tensor/dispatch_stub.hpp"
#include "lamppp/tensor/tensor.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::ops {

/// @internal
using sum_fn = TensorImpl (*)(const TensorImpl&, size_t axis);
using max_fn = TensorImpl (*)(const TensorImpl&, size_t axis);
using min_fn = TensorImpl (*)(const TensorImpl&, size_t axis);
using prod_fn = TensorImpl (*)(const TensorImpl&, size_t axis);

LMP_DECLARE_DISPATCH(sum_fn, sum_stub);
LMP_DECLARE_DISPATCH(max_fn, max_stub);
LMP_DECLARE_DISPATCH(min_fn, min_stub);
LMP_DECLARE_DISPATCH(prod_fn, prod_stub);
/// @endinternal

/**
 * @brief Sum of a tensor
 * @param a The tensor to sum
 * @param axis The axis to sum over
 * @note this function is keep_dims=True. If you want keep_dims=False,
 * you can use squeeze() after this function.
 * @return A new tensor with the result of the sum
 */
Tensor sum(const Tensor& a, size_t axis);

/**
 * @brief Maximum of a tensor
 * @param a The tensor to find the maximum of
 * @param axis The axis to find the maximum of
 * @note this function is keep_dims=True. If you want keep_dims=False,
 * you can use squeeze() after this function.
 * @return A new tensor with the result of the maximum
 */
Tensor max(const Tensor& a, size_t axis);

/**
 * @brief Minimum of a tensor
 * @param a The tensor to find the minimum of
 * @param axis The axis to find the minimum of
 * @note this function is keep_dims=True. If you want keep_dims=False,
 * you can use squeeze() after this function.
 * @return A new tensor with the result of the minimum
 */
Tensor min(const Tensor& a, size_t axis);

/**
 * @brief Product of a tensor
 * @param a The tensor to find the product of
 * @param axis The axis to find the product of
 * @note this function is keep_dims=True. If you want keep_dims=False,
 * you can use squeeze() after this function.
 * @return A new tensor with the result of the product
 */
Tensor prod(const Tensor& a, size_t axis);

}  // namespace lmp::tensor::ops
