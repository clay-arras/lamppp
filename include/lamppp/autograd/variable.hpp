#pragma once

#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>
#include "include/lamppp/tensor/core.hpp"
#include "include/lamppp/tensor/tensor.hpp"
#include "include/lamppp/tensor/tensor_helper.hpp"

namespace lmp::autograd {

class Function;
class Variable;

struct VariableImpl {
  tensor::Tensor data;
  tensor::Tensor grad;
  std::shared_ptr<Function> _grad_fn;
  bool requires_grad;

  explicit VariableImpl(const tensor::Tensor& data, bool requires_grad = false)
      : data(tensor::Tensor(data)),
        grad(zeros_like(data)),
        requires_grad(requires_grad),
        _grad_fn(nullptr) {}
};

class Variable {
 public:
  Variable() = default;
  explicit Variable(const tensor::Tensor& data, bool requires_grad = false)
      : impl_(std::make_shared<VariableImpl>(data, requires_grad)) {}

  const tensor::Tensor& grad() const { return impl_->grad; }
  const tensor::Tensor& data() const { return impl_->data; }
  const std::shared_ptr<Function>& grad_fn() const { return impl_->_grad_fn; }
  const bool requires_grad() const { return impl_->requires_grad; }

  void zero_grad() {
    impl_->grad = zeros_like(impl_->grad);
  }  // TODO: this can be better, implement fill in tensor
  void incr_grad(const tensor::Tensor& other_grad) {
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

struct VariableOpFact {
  template <typename Op, typename... Args>
  static variable_list apply(variable_list variables, Args&&... args) {
    Op op_fn(std::forward<Args>(args)...);
    variable_list result =
        op_fn.template apply<Args...>(variables, std::forward<Args>(args)...);
    return result;
  }
};

}  // namespace lmp::autograd
