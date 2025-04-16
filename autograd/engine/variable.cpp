#include "variable.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <unordered_set>
#include "autograd/engine/functions/basic_ops.h"
#include "autograd/engine/functions/unary_ops.h"
#include "autograd/engine/functions/matrix_ops.h"
#include "autograd/engine/functions/reduct_ops.h"

namespace autograd {

void Variable::backward() {
  std::vector<Variable> topo = topological_sort();
  impl_->grad =
      Tensor(std::vector<float>(impl_->data.data.size(), 1),
             impl_->data.shape);  // TODO(nlin): make this better with getters
  assert(!topo.empty());
  for (Variable& node :
       topo) {  // TODO(nlin): what is this, how to get upstream grad?
    if (node.grad_fn() != nullptr) {
      node.grad_fn()->apply({node});
    }
  }
}

void Variable::dfs(const Variable& v, std::unordered_set<void*>& visited,
                   std::vector<Variable>& topo) const {
  if (visited.find(static_cast<void*>(v.impl_.get())) ==
      visited.end()) {  // TODO(nlin): when would saved_inputs be zero???
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


Variable Variable::operator+(const Variable& other)
    const {  // TODO(nlin): need to optimize s.t. if requires_grad is false then it doesn't do the make_shared
  auto add_fn = std::make_shared<Add>(); // TODO(nlin): need to remove the pointer, maybe make the AddFn static or something
  variable_list result = add_fn->apply({*this, other});
  return result[0];
}

Variable Variable::operator-(const Variable& other) const {
  auto sub_fn = std::make_shared<Subtract>();
  variable_list result = sub_fn->apply({*this, other});
  return result[0];
}

Variable Variable::operator*(const Variable& other) const {
  auto mul_fn = std::make_shared<Multiply>();
  variable_list result = mul_fn->apply({*this, other});
  return result[0];
}

Variable Variable::operator/(const Variable& other) const {
  auto div_fn = std::make_shared<Divide>();
  variable_list result = div_fn->apply({*this, other});
  return result[0];
}

Variable Variable::exp() const {
  auto exp_fn = std::make_shared<Exponential>();
  variable_list result = exp_fn->apply({*this});
  return result[0];
}

Variable Variable::log() const {
  auto log_fn = std::make_shared<Logarithm>();
  variable_list result = log_fn->apply({*this});
  return result[0];
}

Variable Variable::relu() const {
  auto relu_fn = std::make_shared<ReLU>();
  variable_list result = relu_fn->apply({*this});
  return result[0];
}

Variable Variable::matmul(const Variable& other) const {
  auto matmul_fn = std::make_shared<MatrixMultiplication>();
  variable_list result = matmul_fn->apply({*this, other});
  return result[0];
}

Variable Variable::transpose() const {
  auto transpose_fn = std::make_shared<Transpose>();
  variable_list result = transpose_fn->apply({*this});
  return result[0];
}

Variable Variable::sum(int axis) const {
  auto sum_fn = std::make_shared<Summation>(axis);
  variable_list result = sum_fn->apply({*this});
  return result[0];
}

Variable Variable::operator==(const Variable& other) const { // TODO(nlin): separation of responsibility: maybe this is bad and just remove all this and keep it to tensor maybe? 
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

std::ostream& operator<<(std::ostream& os, const Variable& obj) {
  os << "Variable(requires_grad=" << obj.requires_grad();
  os << ", data=" << obj.data();
  os << ", grad=" << obj.grad();
  os << ", grad_fn=" << obj.grad_fn() << ")";
  return os;
}

}