#pragma once

#include "lamppp/tensor/scalar.hpp"  // TODO : maybe move scalar somewhere ?
#include "variable.hpp"

namespace lmp::autograd {

using std::multiplies;

/**
 * @brief Create a variable with all zeros
 * @param shape The shape of the variable
 * @param device The device to create the variable on
 * @param dtype The data type of the variable
 * @param requires_grad Whether the variable requires gradients
 * @return A variable with all zeros
 */
Variable zeros(const std::vector<size_t>& shape, tensor::DeviceType device,
               tensor::DataType dtype, bool requires_grad);

/**
 * @brief Create a variable with all ones
 * @param shape The shape of the variable
 * @param device The device to create the variable on
 * @param dtype The data type of the variable
 * @param requires_grad Whether the variable requires gradients
 * @return A variable with all ones
 */
Variable ones(const std::vector<size_t>& shape, tensor::DeviceType device,
              tensor::DataType dtype, bool requires_grad);

/**
 * @brief Create a variable with random values
 * @param shape The shape of the variable
 * @param device The device to create the variable on
 * @param dtype The data type of the variable
 * @param requires_grad Whether the variable requires gradients
 * @return A variable with random values
 */
Variable rand(const std::vector<size_t>& shape, tensor::DeviceType device,
              tensor::DataType dtype, bool requires_grad);

/// @internal
template <typename>
struct IsVector : std::false_type {};
template <typename U, typename Alloc>
struct IsVector<std::vector<U, Alloc>> : std::true_type {};
/// @endinternal

/// @internal
struct TensorHelper {
  std::vector<tensor::Scalar> data;
  std::vector<size_t> shape;
  template <typename T>
  void unroll(const std::vector<T>& tensor, size_t depth = 0) {
    if (depth >= shape.size()) {
      shape.push_back(tensor.size());
    }
    LMP_CHECK(tensor.size() == shape[depth]) <<
              "Dimensions along axis must be consistent.";
    if constexpr (IsVector<T>::value) {
      for (const T& t : tensor) {
        unroll(t, depth + 1);
      }
    } else {
      data.insert(data.end(), tensor.begin(), tensor.end());
    }
  }
};
/// @endinternal

/// @internal
/**
 * @brief Create a variable from a vector
 * @param data The data to create the variable from. 
 * @param device The device to create the variable on
 * @param dtype The data type of the variable
 * @param requires_grad Whether the variable requires gradients
 * @return A variable with the given data, device, dtype, and requires_grad
 * @details this function can accept a vector of any type that can be converted to a tensor::Scalar
 * and will unroll the vector to a single tensor. e.g. a std::vector<vector<int>> will be unrolled to a single tensor::Tensor
 */
template <typename V>
Variable tensor(const std::vector<V>& data, tensor::DeviceType device,
                tensor::DataType dtype, bool requires_grad) {
  TensorHelper constr;
  constr.unroll(data);
  return Variable(tensor::Tensor(constr.data, constr.shape, device, dtype),
                  requires_grad);
}

}  // namespace lmp::autograd