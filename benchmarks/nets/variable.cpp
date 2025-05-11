#include "variable.h"
#include <memory>
#include "include/lamppp/tensor/functions/basic_ops.h"
#include "include/lamppp/tensor/functions/unary_ops.h"

Variable Variable::operator+(const Variable& other) const {
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
