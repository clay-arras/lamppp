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

void test_autograd_ops() {
  // std::vector<float> a_data = {1.0, 2.0, 3.0, 4.0};
  // std::vector<float> b_data = {0.1, 0.2, 0.3, 0.4};
  // std::vector<int> shape_2x2 = {2, 2};

  // Variable a(Tensor(a_data, shape_2x2), true);
  // Variable b(Tensor(b_data, shape_2x2), true);

  // // Test sum
  // Variable sum_result = a.sum(0);
  // sum_result.backward();
  // std::cout << "Sum gradient:" << std::endl;
  // std::cout << a.grad() << std::endl;
  // a.zero_grad();

  // // Test matmul
  // Variable matmul_result = a.matmul(b);
  // matmul_result.backward();
  // std::cout << "\nMatmul gradient for a:" << std::endl;
  // std::cout << a.grad() << std::endl;
  // std::cout << "\nMatmul gradient for b:" << std::endl;
  // std::cout << b.grad() << std::endl;
  // a.zero_grad();
  // b.zero_grad();

  // // Test transpose
  // Variable transpose_result = a.transpose();
  // transpose_result.backward();
  // std::cout << "\nTranspose gradient:" << std::endl;
  // std::cout << a.grad() << std::endl;
  // a.zero_grad();

  // // Test exp
  // Variable exp_result = a.exp();
  // exp_result.backward();
  // std::cout << "\nExp gradient:" << std::endl;
  // std::cout << a.grad() << std::endl;

  // Create tensors similar to PyTorch example

  std::vector<float> weights1_data = {0.1, 0.2, 0.3, 0.4};
  std::vector<float> inputs_data = {0.05, 0.15, 0.25, 0.35};
  std::vector<float> weights2_data = {0.01, 0.02, 0.03, 0.04};
  std::vector<int> shape_2x2 = {2, 2};

  Variable weights1(Tensor(weights1_data, shape_2x2), true);
  Variable inputs(Tensor(inputs_data, shape_2x2), true);
  Variable weights2(Tensor(weights2_data, shape_2x2), true);

  // Forward pass
  Variable layer1 = weights1.matmul(inputs.transpose());

  Variable layer1_activated = layer1.relu();

  Variable layer2 = layer1_activated.matmul(weights2);
  Variable layer2_sum = layer2.sum(1);
  
  // Create a 1x2 tensor of ones for expansion
  std::vector<float> ones_data = {1.0, 1.0};
  std::vector<int> shape_1x2 = {1, 2};
  Variable ones(Tensor(ones_data, shape_1x2), false);
  
  Variable layer2_expanded = layer2_sum.matmul(ones);
  
  // Element-wise multiplication
  Variable layer3 = layer2_expanded * weights1;
  Variable loss = layer3.sum(0).sum(1);
  
  // Backward pass
  loss.backward();

  std::cout << layer1 << std::endl;
  std::cout << layer2 << std::endl;
  
  std::cout << "\nNeural Network Test Results:" << std::endl;
  std::cout << "Gradient of weights1:" << std::endl;
  std::cout << weights1.grad() << std::endl;

  std::cout << "Gradient of layer1:" << std::endl;
  std::cout << layer1.grad() << std::endl;

  std::cout << "Gradient of layer1_activated:" << std::endl;
  std::cout << layer1_activated.grad() << std::endl;

  std::cout << "Gradient of layer2:" << std::endl;
  std::cout << layer2.grad() << std::endl;

  std::cout << "Gradient of layer2_sum:" << std::endl;
  std::cout << layer2_sum.grad() << std::endl;

  std::cout << "Gradient of layer2_expanded:" << std::endl;
  std::cout << layer2_expanded.grad() << std::endl;

  std::cout << "Gradient of layer3:" << std::endl;
  std::cout << layer3.grad() << std::endl;

}


void test_matmul() {
    std::vector<float> a_data = {1, 2, 3, 4, 5, 6};
    std::vector<int> a_shape = {2, 3}; // 2x3 matrix
    Tensor a(a_data, a_shape);
    
    std::vector<float> b_data = {10, 20, 30}; // Changed to 3x1 matrix
    std::vector<int> b_shape = {3, 1}; // 3x1 matrix
    Tensor b(b_data, b_shape);
    
    // Print original matrices
    std::cout << "Matrix A (2x3):" << std::endl;
    std::cout << a << std::endl;
    
    std::cout << "Matrix B (3x1):" << std::endl; // Updated to reflect new shape
    std::cout << b << std::endl;
    
    // Compute A*B
    Tensor ab = a.matmul(b);
    std::cout << "A*B result:" << std::endl;
    std::cout << ab << std::endl;
    std::cout << "A*B shape should be [2,1]: [" << ab.shape[0] << "," << ab.shape[1] << "]" << std::endl; // Updated expected shape

    // Verify with Eigen directly
    std::cout << "Verifying with Eigen directly..." << std::endl;
    Eigen::Map<Eigen::MatrixXf> eigen_a(a.data.data(), 2, 3);
    Eigen::Map<Eigen::MatrixXf> eigen_b(b.data.data(), 3, 1); // Updated to match new shape
    Eigen::MatrixXf eigen_result = eigen_a * eigen_b;
    std::cout << "Eigen result:" << std::endl;
    std::cout << eigen_result << std::endl;
    
    // Try B*A (which should fail due to dimension mismatch)
    std::cout << "Trying B*A (should fail due to dimension mismatch)..." << std::endl;
    try {
        Tensor ba = b.matmul(a);
        std::cout << "B*A result (should not happen):" << std::endl;
        std::cout << ba << std::endl;
    } catch (const std::exception& e) {
        std::cout << "Error as expected: " << e.what() << std::endl;
    }
    
}



}  // namespace

}  // namespace autograd

int main() {
  // autograd::test_matmul();
  autograd::test_autograd_ops();
  // autograd::test_neural_network();
  return 0;
}
