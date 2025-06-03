#pragma once

#include "lamppp/tensor/device_type.hpp"
#include "lamppp/tensor/dispatch_stub.hpp"
#include "lamppp/tensor/tensor.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::ops {

/// @internal
using matmul_fn = TensorImpl (*)(const TensorImpl&, const TensorImpl&);
using transpose_fn = TensorImpl (*)(const TensorImpl&);

LMP_DECLARE_DISPATCH(matmul_fn, matmul_stub);
LMP_DECLARE_DISPATCH(transpose_fn, transpose_stub);
/// @endinternal

/**
 * @brief Matrix multiplication of two tensors
 * @param a The first tensor
 * @param b The second tensor
 * @return A new tensor with the result of the multiplication
 */
Tensor matmul(const Tensor& a, const Tensor& b);

/**
 * @brief Transpose a tensor
 * @param a The tensor to transpose
 * @return A new tensor with the result of the transpose operation
 * @note unlike Pytorch, this function returns a new tensor, not a view.
 */
Tensor transpose(const Tensor& a);

}  // namespace lmp::tensor::ops
