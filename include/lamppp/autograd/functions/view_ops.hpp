#pragma once

#include "lamppp/autograd/forward_function.hpp"
#include "lamppp/autograd/function.hpp"
#include "lamppp/autograd/functions/overloads.hpp"
#include "lamppp/tensor/device_type.hpp"

namespace lmp::autograd::ops {

/// @internal
struct ReshapeBackward : public Function {
  std::vector<size_t> shape;
  explicit ReshapeBackward(std::vector<size_t> shape)
      : shape(std::move(shape)) {}
  variable_list apply(const variable_list& gradOutputs) override;
};
struct Reshape : public ForwardFunction<Reshape> {
  using DefaultBackward = ReshapeBackward;
  std::vector<size_t> shape;
  explicit Reshape(std::vector<size_t> shape) : shape(std::move(shape)) {}
  tensor::Tensor execute(const variable_list& inputs) const;
};

struct SqueezeBackward : public Function {
  size_t axis;
  explicit SqueezeBackward(size_t axis) : axis(axis) {}
  variable_list apply(const variable_list& gradOutputs) override;
};
struct Squeeze : public ForwardFunction<Squeeze> {
  using DefaultBackward = SqueezeBackward;
  size_t axis;
  explicit Squeeze(size_t axis) : axis(axis) {}
  tensor::Tensor execute(const variable_list& inputs) const;
};

struct ExpandDimsBackward : public Function {
  size_t axis;
  explicit ExpandDimsBackward(size_t axis) : axis(axis) {}
  variable_list apply(const variable_list& gradOutputs) override;
};
struct ExpandDims : public ForwardFunction<ExpandDims> {
  using DefaultBackward = ExpandDimsBackward;
  size_t axis;
  explicit ExpandDims(size_t axis) : axis(axis) {}
  tensor::Tensor execute(const variable_list& inputs) const;
};

struct ToBackward : public Function {
  tensor::DeviceType device;
  explicit ToBackward(tensor::DeviceType device) : device(device) {}
  variable_list apply(const variable_list& gradOutputs) override;
};
struct To : public ForwardFunction<To> {
  using DefaultBackward = ToBackward;
  tensor::DeviceType device;
  explicit To(tensor::DeviceType device) : device(device) {}
  tensor::Tensor execute(const variable_list& inputs) const;
};
/// @endinternal

/**
 * @brief Reshape a variable
 * @param a The variable to reshape
 * @param shape The new shape
 * @return The reshaped variable
 */
inline Variable reshape(const Variable& a, const std::vector<size_t>& shape) {
  return VariableOpFact::apply<Reshape>({a}, shape)[0];
}

/**
 * @brief Squeeze a variable along an axis
 * @param a The variable to squeeze
 * @param axis The axis to squeeze along
 * @return The squeezed variable
 */
inline Variable squeeze(const Variable& a, size_t axis) {
  return VariableOpFact::apply<Squeeze>({a}, axis)[0];
}

/**
 * @brief Expand a variable along an axis
 * @param a The variable to expand
 * @param axis The axis to expand along
 * @return The expanded variable
 */
inline Variable expand_dims(const Variable& a, size_t axis) {
  return VariableOpFact::apply<ExpandDims>({a}, axis)[0];
}

/**
 * @brief Move a variable to a different device
 * @param a The variable to move
 * @param device The device to move the variable to
 * @return The moved variable
 */
inline Variable to(const Variable& a, tensor::DeviceType device) {
  return VariableOpFact::apply<To>({a}, device)[0];
}

}  // namespace lmp::autograd::ops