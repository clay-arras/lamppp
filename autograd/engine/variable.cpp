#include "variable.hpp"
#include <algorithm>
#include <iostream>
#include <memory>
#include <unordered_set>
#include <autograd/engine/forward_function.hpp>
#include <autograd/engine/function.hpp>

namespace autograd {

void Variable::backward() {
  std::vector<Variable> topo = topological_sort();
  impl_->grad.fill(1);
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