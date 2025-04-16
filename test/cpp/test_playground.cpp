#include <iostream>
#include "autograd/engine/tensor.h"
#include "autograd/engine/variable_ops.h"
#include "autograd/engine/variable.h"
#include "autograd/engine/functions/matrix_ops.h"
#include "autograd/engine/functions/reduct_ops.h"

namespace autograd {

namespace {

void test_neural_network() {
  std::vector<float> weights1_data = {0.1, 0.2, 0.3, 0.4};
  std::vector<float> inputs_data = {0.05, 0.15, 0.25, 0.35};
  std::vector<float> weights2_data = {0.01, 0.02, 0.03, 0.04};
  std::vector<float> expand_data = {1.0, 1.0};
  
  std::vector<int> shape_2x2 = {2, 2};
  std::vector<int> shape_1x2 = {1, 2};
  
  Variable weights1(Tensor(weights1_data, shape_2x2), true);
  Variable inputs(Tensor(inputs_data, shape_2x2), true);
  Variable weights2(Tensor(weights2_data, shape_2x2), true);
  Variable expand_tensor(Tensor(expand_data, shape_1x2), false);
  
  const Variable& layer1_linear = weights1.matmul(inputs.transpose());
  const Variable& layer1_activated = layer1_linear.relu();
  
  const Variable& layer2_linear = layer1_activated.matmul(weights2);
  const Variable& layer2_sum = layer2_linear.sum(1);
  const Variable& layer2_output = layer2_sum;
  
  const Variable& layer2_output_expanded = layer2_output.matmul(expand_tensor);
  
  const Variable& layer3_linear = layer2_output_expanded * weights1;
  const Variable& layer3_nonlinear = layer3_linear.exp();
  const Variable& layer3_output = layer3_nonlinear + layer2_output_expanded;
  
  const Variable& logits = layer3_output.matmul(weights2.transpose());
  
  const Variable& probabilities = 1.0F / (1.0F + (-1.0F * logits).exp());
  
  float epsilon = 1e-10;
  const Variable& log_probs = (probabilities + epsilon).log();
  const Variable& neg_log_probs = (-1.0F) * log_probs + 0.1F;

  Variable loss_sum = neg_log_probs.sum(0).sum(1);
  
  std::cout << loss_sum << std::endl;
  loss_sum.backward();
  
  std::cout << "Gradient of weights1:" << std::endl;
  std::cout << weights1.grad() << std::endl;
  
  std::cout << "\nGradient of inputs:" << std::endl;
  std::cout << inputs.grad() << std::endl;
  
  std::cout << "\nGradient of weights2:" << std::endl;
  std::cout << weights2.grad() << std::endl;
}

}  // namespace

}  // namespace autograd

int main() {
  autograd::test_neural_network();
  return 0;
}
