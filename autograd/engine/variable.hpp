#pragma once

#ifndef _VARIABLE_H_
#define _VARIABLE_H_

#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>
#include "tensor.hpp"
#include "tensor_ops.hpp"

namespace autograd {

class Function;
class Variable;

struct VariableImpl {
  Tensor data;
  Tensor grad;
  std::shared_ptr<Function> _grad_fn;
  bool requires_grad;

  explicit VariableImpl(const Tensor& data, bool requires_grad = false) {
    Tensor tmp(data);
    this->grad = std::move(tmp);
    this->grad.fill(0);
    this->data = std::move(data);
    this->requires_grad = requires_grad;
  }
};

class Variable {
 public:
  explicit Variable(std::shared_ptr<VariableImpl>& impl)
      : impl_(std::move(impl)) {}
  explicit Variable(const Tensor& data, bool requires_grad = false)
      : impl_(std::make_shared<VariableImpl>(data, requires_grad)) {}

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
  std::shared_ptr<VariableImpl> impl_;

  std::vector<Variable> topological_sort();
  void dfs(const Variable& v, std::unordered_set<void*>& visited,
           std::vector<Variable>& topo) const;
};

using variable_list = std::vector<Variable>;

}  // namespace autograd

#endif