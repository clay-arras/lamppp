#include "reduct_ops.h"
#include <cassert>
#include "autograd/engine/variable.h"
#include <Eigen/Core>

namespace autograd {

variable_list SummationBackward::apply(const variable_list& gradOutputs) {
  assert(gradOutputs.size() == 1);
  const Variable& grad = gradOutputs[0];
  Variable& self = (*saved_inputs)[0];
  
  std::vector<float> zeros(self.data().data.size(), 0.0F);
  Tensor backgrad(zeros, self.data().shape);

  if (axis == 0) {
    const int rows = self.data().shape[0];
    const int cols = self.data().shape[1];
    
    Eigen::Map<Eigen::MatrixXf> result_mat(backgrad.data.data(), rows, cols);
    Eigen::Map<const Eigen::VectorXf> grad_vec(grad.data().data.data(), cols);
    
    for (int c = 0; c < cols; c++) {
      result_mat.col(c).setConstant(grad_vec(c));
    }
  } else if (axis == 1) {
    const int rows = self.data().shape[0];
    const int cols = self.data().shape[1];
    
    Eigen::Map<Eigen::MatrixXf> result_mat(backgrad.data.data(), rows, cols);
    Eigen::Map<const Eigen::VectorXf> grad_vec(grad.data().data.data(), rows);
    
    for (int r = 0; r < rows; r++) {
      result_mat.row(r).setConstant(grad_vec(r));
    }
  } else {
    assert(0);
  }
  
  self.incr_grad(backgrad);

  variable_list grad_inputs = {grad}; 
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

}
