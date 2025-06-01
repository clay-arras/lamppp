#pragma once

#include "lamppp/autograd/forward_function.hpp"
#include "lamppp/autograd/function.hpp"
#include "lamppp/autograd/functions/overloads.hpp"
#include "lamppp/tensor/device_type.hpp"

namespace lmp::autograd::ops {

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
  tensor::Tensor execute(const variable_list& inputs);
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
  tensor::Tensor execute(const variable_list& inputs);
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
  tensor::Tensor execute(const variable_list& inputs);
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
  tensor::Tensor execute(const variable_list& inputs);
};

inline Variable reshape(const Variable& a, const std::vector<size_t>& shape) {
  return VariableOpFact::apply<Reshape>({a}, shape)[0];
}

inline Variable squeeze(const Variable& a, size_t axis) {
  return VariableOpFact::apply<Squeeze>({a}, axis)[0];
}

inline Variable expand_dims(const Variable& a, size_t axis) {
  return VariableOpFact::apply<ExpandDims>({a}, axis)[0];
}

inline Variable to(const Variable& a, tensor::DeviceType device) {
  return VariableOpFact::apply<To>({a}, device)[0];
}

}  // namespace lmp::autograd::ops