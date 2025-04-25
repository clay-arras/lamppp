#pragma once

#ifndef _VARIABLE_H_
#define _VARIABLE_H_

#include <functional>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>
#include "function.h"
#include "tensor.h"

namespace autograd {

class Function;
class Variable;

struct VariableImpl {
  Tensor data;
  Tensor grad;
  std::shared_ptr<Function> _grad_fn;
  bool requires_grad;

  explicit VariableImpl(Tensor data, bool requires_grad = false) {
    this->data = std::move(data);
    this->grad =
        Tensor(std::vector<float>(this->data.size(), 0.0F), this->data.shape());
    this->requires_grad = requires_grad;
  }
};

struct VariableOpFact {
  template <typename Op, typename... Args>
  static variable_list apply(variable_list variables, Args&&... args) {
    Op op_fn(std::forward<Args>(args)...);
    variable_list result =
        op_fn.template apply<Args...>(variables, std::forward<Args>(args)...);
    return result;
  }
};

class Variable {
 public:
  explicit Variable(std::shared_ptr<VariableImpl>& impl)
      : impl_(std::move(impl)) {}
  explicit Variable(const Tensor& data, bool requires_grad = false)
      : impl_(std::make_shared<VariableImpl>(data, requires_grad)) {}

  std::shared_ptr<VariableImpl> impl_;

  void zero_grad() {
    impl_->grad =
        Tensor(std::vector<float>(data().size(), 0.0F), data().shape());
  }
  void incr_grad(const Tensor& other_grad) {
    impl_->grad = impl_->grad + other_grad;
  }
  void set_grad_fn(std::shared_ptr<Function> grad_fn) {
    impl_->_grad_fn = std::move(grad_fn);
  }

  const Tensor& grad() const { return impl_->grad; }
  const Tensor& data() const { return impl_->data; }
  const std::shared_ptr<Function>& grad_fn() const { return impl_->_grad_fn; }
  bool requires_grad() const { return impl_->requires_grad; }

  void backward();
  std::vector<Variable> topological_sort();

  Variable operator+(const Variable& other) const;
  Variable operator-(const Variable& other) const;
  Variable operator*(const Variable& other) const;
  Variable operator/(const Variable& other) const;

  Variable operator==(const Variable& other) const;
  Variable operator!=(const Variable& other) const;
  Variable operator>=(const Variable& other) const;
  Variable operator<=(const Variable& other) const;
  Variable operator>(const Variable& other) const;
  Variable operator<(const Variable& other) const;

  Variable matmul(const Variable& other) const;
  Variable transpose() const;
  Variable sum(int axis) const;
  Variable max(int axis) const;

  Variable exp() const;
  Variable log() const;
  Variable relu() const;

  friend std::ostream& operator<<(std::ostream& os, const Variable& obj);

 private:
  void dfs(const Variable& v, std::unordered_set<void*>& visited,
           std::vector<Variable>& topo) const;
};

}  // namespace autograd

namespace std {
template <>
struct hash<autograd::Variable> {
  size_t operator()(const autograd::Variable& v) const {
    return std::hash<void*>{}(v.impl_.get());
  }
};
}  // namespace std

#endif