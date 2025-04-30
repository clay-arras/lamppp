#include "variable.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <unordered_set>
#include "autograd/engine/functions/basic_ops.h"
#include "autograd/engine/functions/matrix_ops.h"
#include "autograd/engine/functions/reduct_ops.h"
#include "autograd/engine/functions/unary_ops.h"

namespace autograd {

void Variable::backward() {
  std::vector<Variable> topo = topological_sort();
  // std::cout << "GRAD BEFORE BACK " << impl_->grad << std::endl;
  impl_->grad.fill(1);
  // std::cout << "GRAD AFTER BACK " << impl_->grad << std::endl;
      // Tensor(std::vector<float>(impl_->data.size(), 1), impl_->data.shape());
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

Variable Variable::operator+(const Variable& other) const {
  return VariableOpFact::apply<Add>({*this, other})[0];
}

Variable Variable::operator-(const Variable& other) const {
  return VariableOpFact::apply<Subtract>({*this, other})[0];
}

Variable Variable::operator*(const Variable& other) const {
  return VariableOpFact::apply<Multiply>({*this, other})[0];
}

Variable Variable::operator/(const Variable& other) const {
  return VariableOpFact::apply<Divide>({*this, other})[0];
}

Variable Variable::exp() const {
  return VariableOpFact::apply<Exponential>({*this})[0];
}

Variable Variable::log() const {
  return VariableOpFact::apply<Logarithm>({*this})[0];
}

Variable Variable::relu() const {
  return VariableOpFact::apply<ReLU>({*this})[0];
}

Variable Variable::matmul(const Variable& other) const {
  return VariableOpFact::apply<MatrixMultiplication>({*this, other})[0];
}

Variable Variable::transpose() const {
  return VariableOpFact::apply<Transpose>({*this})[0];
}

Variable Variable::sum(int axis) const {
  return VariableOpFact::apply<Summation>({*this}, axis)[0];
}

Variable Variable::max(int axis) const {
  return VariableOpFact::apply<Maximum>({*this}, axis)[0];
}

Variable Variable::operator==(const Variable& other)
    const {  // TODO(nlin): separation of responsibility: maybe this is bad and just remove all this and keep it to tensor maybe?
  return Variable(data() == other.data(), false);
}

Variable Variable::operator!=(const Variable& other) const {
  return Variable(data() != other.data(), false);
}

Variable Variable::operator>=(const Variable& other) const {
  return Variable(data() >= other.data(), false);
}

Variable Variable::operator<=(const Variable& other) const {
  return Variable(data() <= other.data(), false);
}

Variable Variable::operator>(const Variable& other) const {
  return Variable(data() > other.data(), false);
}

Variable Variable::operator<(const Variable& other) const {
  return Variable(data() < other.data(), false);
}

}  // namespace autograd