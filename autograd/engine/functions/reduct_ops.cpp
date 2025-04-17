#include "reduct_ops.h"
#include <Eigen/Core>
#include <cassert>
#include "autograd/engine/variable.h"

namespace autograd {

variable_list SummationBackward::apply(const variable_list& gradOutputs) { // TODO(nlin): need to make this more general, this is disgusting
  assert(gradOutputs.size() == 1);
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];

  std::vector<float> zeros(self.data().size(), 0.0F);
  Tensor backgrad(zeros, self.data().shape);

  if (axis == 0) {
    const int rows = self.data().shape[0];
    const int cols = self.data().shape[1];

    Eigen::Map<Eigen::MatrixXf> result_mat(backgrad.data.data(), rows, cols);
    Eigen::Map<const Eigen::VectorXf> grad_vec(grad.grad().data.data(), cols);

    for (int c = 0; c < cols; c++) {
      result_mat.col(c).setConstant(grad_vec(c));
    }
  } else if (axis == 1) {
    const int rows = self.data().shape[0];
    const int cols = self.data().shape[1];

    Eigen::Map<Eigen::MatrixXf> result_mat(backgrad.data.data(), rows, cols);
    Eigen::Map<const Eigen::VectorXf> grad_vec(grad.grad().data.data(), rows);

    for (int r = 0; r < rows; r++) {
      result_mat.row(r).setConstant(grad_vec(r));
    }
  } else {
    assert(0);
  }

  self.incr_grad(backgrad);

  variable_list grad_inputs = {};
  return grad_inputs;
}

variable_list MaximumBackward::apply(const variable_list& gradOutputs) { // TODO(nlin): this is disgusting + slow, values can be cached
  assert(gradOutputs.size() == 1);
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];

  std::vector<float> zeros(self.data().size(), 0.0F);
  Tensor backgrad(zeros, self.data().shape);
  Tensor mask(zeros, self.data().shape);
  Tensor max_mask = self.data().max(axis);

  if (axis == 0) {
    const int rows = self.data().shape[0];
    const int cols = self.data().shape[1];

    Eigen::Map<Eigen::MatrixXf> result_mat(backgrad.data.data(), rows, cols);
    Eigen::Map<Eigen::MatrixXf> mask_mat(mask.data.data(), rows, cols);
    Eigen::Map<const Eigen::VectorXf> grad_vec(grad.grad().data.data(), cols);
    Eigen::Map<const Eigen::VectorXf> max_vec(max_mask.data.data(), cols);

    for (int c = 0; c < cols; c++) {
      result_mat.col(c).setConstant(grad_vec(c));
      mask_mat.col(c).setConstant(max_vec(c));
    }
  } else if (axis == 1) {
    const int rows = self.data().shape[0];
    const int cols = self.data().shape[1];

    Eigen::Map<Eigen::MatrixXf> result_mat(backgrad.data.data(), rows, cols);
    Eigen::Map<Eigen::MatrixXf> mask_mat(mask.data.data(), rows, cols);
    Eigen::Map<const Eigen::VectorXf> grad_vec(grad.grad().data.data(), rows);
    Eigen::Map<const Eigen::VectorXf> max_vec(max_mask.data.data(), rows);

    for (int r = 0; r < rows; r++) {
      result_mat.row(r).setConstant(grad_vec(r));
      mask_mat.row(r).setConstant(max_vec(r));
    }
  } else {
    assert(0);
  }

  Tensor bool_mask = (mask == self.data());
  self.incr_grad(backgrad * bool_mask);

  variable_list grad_inputs = {};
  return grad_inputs;
}

variable_list Summation::apply(const variable_list& inputs) {
  assert(inputs.size() == 1);
  const Variable& self = inputs[0];

  Variable result = Variable(self.data().sum(axis), true);
  auto backward_fn = std::make_shared<SummationBackward>(axis);
  backward_fn->saved_inputs =
      std::make_unique<variable_list>(variable_list{self});
  backward_fn->axis = axis;
  result.set_grad_fn(backward_fn);
  return {result};
}

variable_list Maximum::apply(const variable_list& inputs) {
  assert(inputs.size() == 1);
  const Variable& self = inputs[0];

  Variable result = Variable(self.data().max(axis), true);
  auto backward_fn = std::make_shared<MaximumBackward>(axis);
  backward_fn->saved_inputs =
      std::make_unique<variable_list>(variable_list{self});
  backward_fn->axis = axis;
  result.set_grad_fn(backward_fn);
  return {result};
}

}  // namespace autograd
