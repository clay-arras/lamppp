#include "constructor.hpp"
#include <numeric>
#include "autograd/engine/variable.hpp"

namespace autograd {

inline namespace functional {
using std::multiplies;

Variable zeros(const std::vector<size_t>& shape, bool requires_grad) {
  size_t sz = std::accumulate(shape.begin(), shape.end(), 1, multiplies<>());
  return Variable(Tensor(std::vector<float>(sz, 0.0F), shape), requires_grad);
}

Variable ones(const std::vector<size_t>& shape, bool requires_grad) {
  size_t sz = std::accumulate(shape.begin(), shape.end(), 1, multiplies<>());
  return Variable(Tensor(std::vector<float>(sz, 1.0F), shape), requires_grad);
}

Variable rand(const std::vector<size_t>& shape, bool requires_grad) {
  size_t sz = std::accumulate(shape.begin(), shape.end(), 1, multiplies<>());
  std::vector<float> rand_vec(sz);
  Eigen::Map<Eigen::ArrayXXf> res(rand_vec.data(), sz, 1);
  res.setRandom();
  return Variable(Tensor(rand_vec, shape), requires_grad);
}

}  // namespace functional

}  // namespace autograd