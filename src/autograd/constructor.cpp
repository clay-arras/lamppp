#include "include/lamppp/autograd/constructor.hpp"
#include <algorithm>
#include <numeric>
#include <random>

namespace autograd {

inline namespace functional {

Variable zeros(const std::vector<size_t>& shape, DeviceType device,
               DataType dtype, bool requires_grad) {
  size_t sz = shape.empty() ? 0
                            : std::accumulate(shape.begin(), shape.end(), 1,
                                              std::multiplies<>());
  return Variable(Tensor(std::vector<Scalar>(sz, 0.0), shape, device, dtype),
                  requires_grad);
}

Variable ones(const std::vector<size_t>& shape, DeviceType device,
              DataType dtype, bool requires_grad) {
  size_t sz = shape.empty() ? 0
                            : std::accumulate(shape.begin(), shape.end(), 1,
                                              std::multiplies<>());
  return Variable(Tensor(std::vector<Scalar>(sz, 1.0), shape, device, dtype),
                  requires_grad);
}

Variable rand(const std::vector<size_t>& shape, DeviceType device,
              DataType dtype, bool requires_grad) {
  size_t sz = shape.empty() ? 0
                            : std::accumulate(shape.begin(), shape.end(), 1,
                                              std::multiplies<>());
  std::vector<Scalar> rand_vec(sz);
  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_real_distribution<> distrib(0.0, 1.0);
  std::generate(rand_vec.begin(), rand_vec.end(),
                [&]() { return distrib(gen); });

  return Variable(Tensor(rand_vec, shape, device, dtype), requires_grad);
}

}  // namespace functional

}  // namespace autograd