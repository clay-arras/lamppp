#pragma once
#include <vector>
#ifndef _VARIABLE_H_
#define _VARIABLE_H_

#include <functional>
#include <memory>
#include <unordered_set>
#include <utility>
#include "function.h"
#include "tensor.h"

class Function;

struct VariableImpl {
  Tensor data;
  Tensor grad;
  std::shared_ptr<Function> _grad_fn;
  bool requires_grad;

  explicit VariableImpl(Tensor data, bool requires_grad = false) {
    this->data = std::move(data);
    this->grad = Tensor(std::vector<float>(this->data.data.size(), 0.0F),
                        this->data.shape);
    this->requires_grad = requires_grad;
  }
};

class Variable {
 public:
  Variable()
      : impl_(std::make_shared<VariableImpl>(Tensor({0.0F}, {1}), false)) {}
  explicit Variable(std::shared_ptr<VariableImpl>& impl)
      : impl_(std::move(impl)) {}
  explicit Variable(const Tensor& data, bool requires_grad = false)
      : impl_(std::make_shared<VariableImpl>(data, requires_grad)) {}

  std::shared_ptr<VariableImpl> impl_;
  Tensor& grad() { return impl_->grad; }
  void zero_grad() {
    impl_->grad =
        Tensor(std::vector<float>(data().data.size(), 0.0F), data().shape);
  }
  void incr_grad(const Tensor& other_grad) {
    impl_->grad = impl_->grad + other_grad;
  }

  Tensor& data() { return impl_->data; }
  std::shared_ptr<Function>& grad_fn() { return impl_->_grad_fn; }
  void set_grad_fn(std::shared_ptr<Function> grad_fn) {
    impl_->_grad_fn = std::move(grad_fn);
  }
  bool& requires_grad() { return impl_->requires_grad; }

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
  bool operator==(const Variable& other) const;

  Variable operator+(float other) const;
  Variable operator-(float other) const;
  Variable operator*(float other) const;
  Variable operator/(float other) const;

  Variable exp() const;
  Variable log() const;
  Variable relu() const;

  friend std::ostream& operator<<(std::ostream& os, const Variable& obj);

 private:
  void dfs(const Variable& v, std::unordered_set<Variable>& visited,
           std::vector<Variable>& topo) const;
};

namespace std {
template <>
struct hash<Variable> {
  size_t operator()(const Variable& v) const {
    return std::hash<void*>{}(v.impl_.get());
  }
};
}  // namespace std

#endif