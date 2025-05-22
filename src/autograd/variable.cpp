#include "lamppp/autograd/variable.hpp"
#include <algorithm>
#include <iostream>
#include <memory>
#include <unordered_set>
#include "lamppp/autograd/function.hpp"
#include "lamppp/common/assert.hpp"

namespace lmp::autograd {

const tensor::Tensor& Variable::grad() const noexcept {
  return impl_->grad;
}
const tensor::Tensor& Variable::data() const noexcept {
  return impl_->data;
}
const std::shared_ptr<Function>& Variable::grad_fn() const noexcept {
  return impl_->_grad_fn;
}
const bool Variable::requires_grad() const noexcept {
  return impl_->requires_grad;
}

void Variable::zero_grad() {
  impl_->grad = zeros_like(impl_->grad);
}  // TODO: this can be better, implement fill in tensor
void Variable::incr_grad(const tensor::Tensor& other_grad) {
  LMP_INTERNAL_ASSERT(other_grad.shape() == impl_->grad.shape(),
                      "There should be no broadcasting in incr_grad");
  impl_->grad = impl_->grad + other_grad;
}
void Variable::set_grad_fn(std::shared_ptr<Function> grad_fn) {
  impl_->_grad_fn = std::move(grad_fn);
}

void Variable::backward() {
  LMP_CHECK(requires_grad(), "Must be declared with requires_grad");
  std::vector<Variable> topo = topological_sort();
  impl_->grad = ones_like(impl_->grad);
  for (Variable& node : topo) {
    if (node.grad_fn() != nullptr) {
      node.grad_fn()->apply({node});
    }
  }
}

void Variable::dfs(const Variable& v, std::unordered_set<void*>& visited,
                   std::vector<Variable>& topo) const {
  if (visited.find(static_cast<void*>(v.impl_.get())) == visited.end()) {
    visited.insert(static_cast<void*>(v.impl_.get()));
    if (v.grad_fn() == nullptr || v.grad_fn()->saved_inputs == nullptr) {
      topo.push_back(v);
      return;
    }
    for (const auto& child : *(v.grad_fn()->saved_inputs)) {
      dfs(child, visited, topo);
    }
    topo.push_back(v);
  }
}

std::vector<Variable> Variable::topological_sort() {
  std::unordered_set<void*> visited;
  std::vector<Variable> topo;

  dfs(*this, visited, topo);
  std::reverse(topo.begin(), topo.end());
  return topo;
}

std::ostream& operator<<(std::ostream& os, const Variable& obj) {
  os << "Variable(requires_grad=" << obj.requires_grad();
  os << ", data=" << obj.data();
  os << ", grad=" << obj.grad();
  os << ", grad_fn=" << obj.grad_fn() << ")";
  return os;
}

}  // namespace lmp::autograd