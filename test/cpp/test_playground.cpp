#include <iostream>
#include "autograd/engine/tensor.h"
#include "autograd/engine/tensor_ops.h"
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
  Variable weights2(Tensor(weights2_data, shape_2x2), true);
  Variable inputs(Tensor(inputs_data, shape_2x2), true);
  Variable expand_tensor(Tensor(expand_data, shape_1x2), false);
  
  for (int i = 0; i < 10; i++) {
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

    float learning_rate = 0.0001;
    weights1 = Variable(weights1.data() - learning_rate * weights1.grad(), true);
    weights2 = Variable(weights2.data() - learning_rate * weights2.grad(), true);
  }
}

void test_autograd_ops() {
  std::vector<float> weights1_data = {0.1, 0.2, 0.3, 0.4};
  std::vector<float> inputs_data = {0.05, 0.15, 0.25, 0.35};
  std::vector<float> weights2_data = {0.01, 0.02, 0.03, 0.04};
  std::vector<int> shape_2x2 = {2, 2};

  Variable weights1(Tensor(weights1_data, shape_2x2), true);
  Variable inputs(Tensor(inputs_data, shape_2x2), true);
  Variable weights2(Tensor(weights2_data, shape_2x2), true);

  Variable layer1 = weights1.matmul(inputs.transpose());
  Variable layer1_activated = layer1.relu();

  Variable layer2 = layer1_activated.matmul(weights2);
  Variable layer2_sum = layer2.sum(1);
  
  std::vector<float> ones_data = {1.0, 1.0};
  std::vector<int> shape_1x2 = {1, 2};
  Variable ones(Tensor(ones_data, shape_1x2), false);
  
  Variable layer2_expanded = layer2_sum.matmul(ones);
  
  Variable layer3 = layer2_expanded * weights1;
  Variable loss = layer3.sum(0).sum(1);

  loss.backward();
  
  std::cout << "\nNeural Network Test Results:" << std::endl;
  std::cout << "Weights1:" << std::endl;
  std::cout << weights1 << std::endl;

  std::cout << "Layer1:" << std::endl;
  std::cout << layer1 << std::endl;

  std::cout << "Layer1 Activated:" << std::endl;
  std::cout << layer1_activated << std::endl;

  std::cout << "Layer2:" << std::endl;
  std::cout << layer2 << std::endl;

  std::cout << "Layer2 Sum:" << std::endl;
  std::cout << layer2_sum << std::endl;

  std::cout << "Layer2 Expanded:" << std::endl;
  std::cout << layer2_expanded << std::endl;

  std::cout << "Layer3:" << std::endl;
  std::cout << layer3 << std::endl;
}


void test_matmul() {
  std::vector<float> weights1_data = {0.1, 0.2, 0.3, 0.4};
  std::vector<float> inputs_data = {0.05, 0.15, 0.25, 0.35};
  std::vector<float> weights2_data = {0.01, 0.02, 0.03, 0.04};
  std::vector<int> shape_2x2 = {2, 2};

  Variable weights1(Tensor(weights1_data, shape_2x2), true);
  Variable inputs(Tensor(inputs_data, shape_2x2), true);
  Variable weights2(Tensor(weights2_data, shape_2x2), true);

  std::cout << "WEIGHTS1: " << weights1 << std::endl;
  std::cout << "INPUT: " << inputs.transpose() << std::endl;
  Variable layer1 = weights1.matmul(inputs.transpose());
  std::cout << "LAYER1: " << layer1 << std::endl;

  std::cout << "---" << std::endl;
  std::cout << "LAYER1: " << layer1 << std::endl;
  std::cout << "WEIGHTS2: " << weights2 << std::endl;
  Variable layer2 = layer1.matmul(weights2);
  std::cout << "LAYER2: " << layer2 << std::endl;
}

}  // namespace

}  // namespace autograd

int main() {
  // autograd::test_matmul();
  // autograd::test_autograd_ops();
  autograd::test_neural_network();
  return 0;
}
