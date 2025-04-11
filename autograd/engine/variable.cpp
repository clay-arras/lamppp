#include "variable.h"
#include "autograd/engine/functions/basic_ops.h"
#include "autograd/engine/functions/unary_ops.h"
#include <memory>
#include <unordered_set>
#include <algorithm>

Variable Variable::operator+(const Variable& other) const { // TODO(nlin): need to optimize s.t. if requires_grad is false then it doesn't do the make_shared
    auto add_fn = std::make_shared<Add>();
    variable_list result = add_fn->apply({*this, other});
    result[0].grad_fn() = add_fn;
    return result[0];
}

Variable Variable::operator-(const Variable& other) const {
    auto sub_fn = std::make_shared<Subtract>();
    variable_list result = sub_fn->apply({*this, other});
    result[0].grad_fn() = sub_fn;
    return result[0];
}

Variable Variable::operator*(const Variable& other) const {
    auto mul_fn = std::make_shared<Multiply>();
    variable_list result = mul_fn->apply({*this, other});
    result[0].grad_fn() = mul_fn;
    return result[0];
}

Variable Variable::operator/(const Variable& other) const {
    auto div_fn = std::make_shared<Divide>();
    variable_list result = div_fn->apply({*this, other});
    result[0].grad_fn() = div_fn;
    return result[0];
}

bool Variable::operator==(const Variable& other) const {
    return impl_ == other.impl_;
} // TODO(nlin): implement broadcasting later

Variable Variable::operator+(float other) const {
    return *this + Variable(Tensor(std::vector<float>(impl_->data.data.size(), other), impl_->data.shape));
}

Variable Variable::operator-(float other) const {
    return *this - Variable(Tensor(std::vector<float>(impl_->data.data.size(), other), impl_->data.shape));
}

Variable Variable::operator*(float other) const {
    return *this * Variable(Tensor(std::vector<float>(impl_->data.data.size(), other), impl_->data.shape));
}

Variable Variable::operator/(float other) const {
    return *this / Variable(Tensor(std::vector<float>(impl_->data.data.size(), other), impl_->data.shape));
}

Variable Variable::exp() const {
    auto exp_fn = std::make_shared<Exponential>();
    variable_list result = exp_fn->apply({*this});
    result[0].grad_fn() = exp_fn;
    return result[0];
}

Variable Variable::log() const {
    auto log_fn = std::make_shared<Logarithm>();
    variable_list result = log_fn->apply({*this});
    result[0].grad_fn() = log_fn;
    return result[0];
}

Variable Variable::relu() const {
    auto relu_fn = std::make_shared<ReLU>();
    variable_list result = relu_fn->apply({*this});
    result[0].grad_fn() = relu_fn;
    return result[0];
}

void Variable::backward() {
  std::vector<Variable> topo = topological_sort();
  impl_->grad = Tensor(std::vector<float>(impl_->data.data.size(), 1), impl_->data.shape); // make this better with getters
  for (Variable& node : topo) {
    node.backward();
  }
}

void Variable::dfs(const Variable& v, std::unordered_set<Variable>& visited, std::vector<Variable>& topo) const {
  if (visited.find(v) == visited.end()) {
    visited.insert(v);
    if (v.grad_fn() == nullptr || v.grad_fn()->saved_inputs == nullptr) { // when would saved_inputs be zero???
        return;
    }
    for (const auto& child : *(v.grad_fn()->saved_inputs)) {
      dfs(child, visited, topo);
    }
    topo.push_back(v);
  }
}

std::vector<Variable> Variable::topological_sort() {
  std::unordered_set<Variable> visited;
  std::vector<Variable> topo;

  dfs(*this, visited, topo);
  std::reverse(topo.begin(), topo.end());
  return topo;
}