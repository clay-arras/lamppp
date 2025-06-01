#include <vector>
#include "lamppp/lamppp.hpp"
#include "lamppp/tensor/device_type.hpp"

int main() {
  auto a = lmp::autograd::Variable(lmp::tensor::Tensor(std::vector<float>{1, 2, 3, 4, 5, 2},
                                                       std::vector<size_t>{3, 2},
                                                       lmp::tensor::DeviceType::CUDA, 
                                                       lmp::tensor::DataType::Float32), true);
  auto b = lmp::autograd::Variable(lmp::tensor::Tensor(std::vector<float>{1, 2, 3.2, -1, 2, 3},
                                                       std::vector<size_t>{3, 2},
                                                       lmp::tensor::DeviceType::CPU, 
                                                       lmp::tensor::DataType::Float32), true);

  auto prod_result = lmp::autograd::ops::prod(a, 1);
  std::cout << "prod result: " << prod_result.data() << std::endl;
  prod_result.backward();
  std::cout << "a grad after prod: " << a.grad() << std::endl;

  a.zero_grad();
  b.zero_grad();

  auto neg_result = lmp::autograd::ops::neg(b);
  std::cout << "neg result: " << neg_result.data() << std::endl;
  neg_result.backward();
  std::cout << "b grad after neg: " << b.grad() << std::endl;

  a.zero_grad();
  b.zero_grad();

  auto c = lmp::autograd::ops::pow(a, -(lmp::autograd::ops::to(b, lmp::tensor::DeviceType::CUDA)));
  std::cout << "pow result: " << c.data() << std::endl;
  c.backward();
  std::cout << "a grad after pow: " << a.grad() << std::endl;
  std::cout << "b grad after pow: " << b.grad() << std::endl;
}