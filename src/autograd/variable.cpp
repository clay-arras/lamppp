#include "include/lamppp/autograd/variable.hpp"
#include <algorithm>
#include <include/lamppp/autograd/forward_function.hpp>
#include <include/lamppp/autograd/function.hpp>
#include <iostream>
#include <memory>
#include <unordered_set>

namespace autograd {

void Variable::backward() {
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

}  // namespace autograd