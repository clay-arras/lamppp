#include "lamppp/autograd/utils/constructor.hpp"
#include <algorithm>
#include <numeric>
#include <random>

namespace lmp::autograd {

Variable zeros(const std::vector<size_t>& shape, bool requires_grad,
               tensor::DeviceType device, tensor::DataType dtype) {
  size_t sz = shape.empty() ? 0
                            : std::accumulate(shape.begin(), shape.end(), 1,
                                              std::multiplies<>());
  return Variable(tensor::Tensor(std::vector<tensor::Scalar>(sz, 0.0), shape,
                                 device, dtype),
                  requires_grad);
}

Variable ones(const std::vector<size_t>& shape, bool requires_grad,
              tensor::DeviceType device, tensor::DataType dtype) {
  size_t sz = shape.empty() ? 0
                            : std::accumulate(shape.begin(), shape.end(), 1,
                                              std::multiplies<>());
  return Variable(tensor::Tensor(std::vector<tensor::Scalar>(sz, 1.0), shape,
                                 device, dtype),
                  requires_grad);
}

Variable rand(const std::vector<size_t>& shape, bool requires_grad,
              tensor::DeviceType device, tensor::DataType dtype) {
  size_t sz = shape.empty() ? 0
                            : std::accumulate(shape.begin(), shape.end(), 1,
                                              std::multiplies<>());
  std::vector<tensor::Scalar> rand_vec(sz);
  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_real_distribution<> distrib(0.0, 1.0);
  std::ranges::generate(rand_vec, [&]() { return distrib(gen); });

  return Variable(tensor::Tensor(rand_vec, shape, device, dtype),
                  requires_grad);
}

Variable randn(tensor::Scalar mean, tensor::Scalar var,
               const std::vector<size_t>& shape, bool requires_grad,
               tensor::DeviceType device, tensor::DataType dtype) {
  size_t sz = shape.empty() ? 0
                            : std::accumulate(shape.begin(), shape.end(), 1,
                                              std::multiplies<>());
  std::vector<tensor::Scalar> rand_vec(sz);

  std::mt19937_64 rg{std::random_device{}()};
  std::normal_distribution<> gauss(mean, var);

  std::ranges::generate(rand_vec, [&]() { return gauss(rg); });
  return Variable(tensor::Tensor(rand_vec, shape, device, dtype),
                  requires_grad);
}

}  // namespace lmp::autograd