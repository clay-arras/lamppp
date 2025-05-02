#include "reduct_ops.hpp"
#include <cassert>
#include "autograd/engine/variable.hpp"

namespace autograd {

variable_list SummationBackward::apply(
    const variable_list&
        gradOutputs) {  // TODO(nlin): need to make this more general, this is disgusting
  // assert(gradOutputs.size() == 1);
  // const Variable& grad = gradOutputs[0];
  // Variable& self = (*saved_inputs)[0];

  // std::vector<float> zeros(self.data().size(), 0.0F);
  // Tensor backgrad(zeros, self.data().shape());

  // if (axis == 0) {
  //   const size_t rows = self.data().shape()[0];
  //   const size_t cols = self.data().shape()[1];

  //   Eigen::Map<Eigen::MatrixXf> result_mat(backgrad.data_ptr<float>(), rows,
  //                                          cols);
  //   Eigen::Map<const Eigen::VectorXf> grad_vec(grad.grad().data<float>().data(),
  //                                              cols);

  //   for (size_t c = 0; c < cols; c++) {
  //     result_mat.col(c).setConstant(grad_vec(c));
  //   }
  // } else if (axis == 1) {
  //   const size_t rows = self.data().shape()[0];
  //   const size_t cols = self.data().shape()[1];

  //   Eigen::Map<Eigen::MatrixXf> result_mat(backgrad.data_ptr<float>(), rows,
  //                                          cols);
  //   Eigen::Map<const Eigen::VectorXf> grad_vec(grad.grad().data<float>().data(),
  //                                              rows);

  //   for (size_t r = 0; r < rows; r++) {
  //     result_mat.row(r).setConstant(grad_vec(r));
  //   }
  // } else {
  //   assert(0);
  // }

  // self.incr_grad(backgrad);

  variable_list grad_inputs = {};
  return grad_inputs;
}

variable_list MaximumBackward::apply(
    const variable_list&
        gradOutputs) {  // TODO(nlin): this is disgusting + slow, values can be cached
  // assert(gradOutputs.size() == 1);
  // const Variable& grad = gradOutputs[0];
  // Variable& self = (*saved_inputs)[0];

  // std::vector<float> zeros(self.data().size(), 0.0F);
  // Tensor backgrad(zeros, self.data().shape());
  // Tensor mask(zeros, self.data().shape());
  // Tensor max_mask = self.data().max(axis);

  // if (axis == 0) {
  //   const size_t rows = self.data().shape()[0];
  //   const size_t cols = self.data().shape()[1];

  //   Eigen::Map<Eigen::MatrixXf> result_mat(backgrad.data().data(), rows, cols);
  //   Eigen::Map<Eigen::MatrixXf> mask_mat(mask.data().data(), rows, cols);
  //   Eigen::Map<const Eigen::VectorXf> grad_vec(grad.grad().data().data(), cols);
  //   Eigen::Map<const Eigen::VectorXf> max_vec(max_mask.data().data(), cols);

  //   for (size_t c = 0; c < cols; c++) {
  //     result_mat.col(c).setConstant(grad_vec(c));
  //     mask_mat.col(c).setConstant(max_vec(c));
  //   }
  // } else if (axis == 1) {
  //   const size_t rows = self.data().shape()[0];
  //   const size_t cols = self.data().shape()[1];

  //   Eigen::Map<Eigen::MatrixXf> result_mat(backgrad.data().data(), rows, cols);
  //   Eigen::Map<Eigen::MatrixXf> mask_mat(mask.data().data(), rows, cols);
  //   Eigen::Map<const Eigen::VectorXf> grad_vec(grad.grad().data().data(), rows);
  //   Eigen::Map<const Eigen::VectorXf> max_vec(max_mask.data().data(), rows);

  //   for (size_t r = 0; r < rows; r++) {
  //     result_mat.row(r).setConstant(grad_vec(r));
  //     mask_mat.row(r).setConstant(max_vec(r));
  //   }
  // } else {
  //   assert(0);
  // }

  // Tensor bool_mask = (mask == self.data());
  // self.incr_grad(backgrad * bool_mask);

  variable_list grad_inputs = {};
  return grad_inputs;
}

Tensor Summation::execute(const variable_list& inputs) const {
  assert(inputs.size() == 1);
  const Variable& self = inputs[0];

  return sum(self.data(), axis);
}

Tensor Maximum::execute(const variable_list& inputs) const {
  assert(inputs.size() == 1);
  const Variable& self = inputs[0];

  return max(self.data(), axis);
}

}  // namespace autograd
