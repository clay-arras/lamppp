#pragma once

#ifndef _VARIABLE_H_
#define _VARIABLE_H_

#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>
#include "tensor.hpp"
#include "tensor_helper.hpp"

namespace autograd {

class Function;
class Variable;

struct VariableImpl {
  Tensor data;
  Tensor grad;
  std::shared_ptr<Function> _grad_fn;
  bool requires_grad;

  VariableImpl() = default;
  explicit VariableImpl(const Tensor& data, bool requires_grad = false) {
    this->data = Tensor(data);
    this->grad = zeros_like(data);
    this->requires_grad = requires_grad;
  }

  ~VariableImpl() = default;
  VariableImpl(const VariableImpl&) = default;
  VariableImpl& operator=(const VariableImpl&) = default;
  VariableImpl(VariableImpl&&) = default;
  VariableImpl& operator=(VariableImpl&&) = default;
};

class Variable {
 public:
  explicit Variable(const Tensor& data, bool requires_grad = false)
      : impl_(std::make_unique<VariableImpl>(data, requires_grad)) {}

  ~Variable() = default;
  Variable(const Variable& other)
      : impl_(std::make_unique<VariableImpl>(*other.impl_)) {}
  Variable& operator=(const Variable& other) {
    if (this != &other) {
      impl_ = std::make_unique<VariableImpl>(*other.impl_);
    }
    return *this;
  }
  Variable(Variable&& other) noexcept = default;
  Variable& operator=(Variable&& other) noexcept = default;

  const Tensor& grad() const { return impl_->grad; }
  const Tensor& data() const { return impl_->data; }
  const std::shared_ptr<Function>& grad_fn() const { return impl_->_grad_fn; }
  const bool requires_grad() const { return impl_->requires_grad; }

  void zero_grad() { impl_->grad.fill(0.0); }
  void incr_grad(const Tensor& other_grad) {
    impl_->grad = impl_->grad + other_grad;
  }
  void set_grad_fn(std::shared_ptr<Function> grad_fn) {
    impl_->_grad_fn = std::move(grad_fn);
  }

  void backward();
  friend std::ostream& operator<<(std::ostream& os, const Variable& obj);

 private:
  std::unique_ptr<VariableImpl> impl_;
  std::vector<Variable> topological_sort();
  void dfs(const Variable& v, std::unordered_set<void*>& visited,
           std::vector<Variable>& topo) const;
};

using variable_list = std::vector<Variable>;

}  // namespace autograd

#endif