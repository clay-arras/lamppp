#pragma once

#include <algorithm>
#include <cassert>
#include "include/lamppp/tensor/scalar.hpp"  // TODO : maybe move scalar somewhere ?
#include "variable.hpp"

namespace lmp::autograd {

inline namespace functional {

using std::multiplies;

Variable zeros(const std::vector<size_t>& shape, tensor::DeviceType device,
               tensor::DataType dtype, bool requires_grad);

Variable ones(const std::vector<size_t>& shape, tensor::DeviceType device,
              tensor::DataType dtype, bool requires_grad);

Variable rand(const std::vector<size_t>& shape, tensor::DeviceType device,
              tensor::DataType dtype, bool requires_grad);

template <typename>
struct IsVector : std::false_type {};
template <typename U, typename Alloc>
struct IsVector<std::vector<U, Alloc>> : std::true_type {};

struct TensorHelper {
  std::vector<tensor::Scalar> data;
  std::vector<size_t> shape;
  template <class T>
  void unroll(const std::vector<T>& tensor, size_t depth = 0) {
    if (depth >= shape.size()) {
      shape.push_back(tensor.size());
    }
    assert(tensor.size() == shape[depth]);
    if constexpr (IsVector<T>::value) {
      for (const T& t : tensor) {
        unroll(t, depth + 1);
      }
    } else {
      data.insert(data.end(), tensor.begin(), tensor.end());
    }
  }
};

template <typename V>
Variable tensor(const std::vector<V>& data, tensor::DeviceType device,
                tensor::DataType dtype, bool requires_grad) {
  TensorHelper constr;
  constr.unroll(data);
  std::vector<tensor::DataType> body(constr.data.size());
  std::transform(__pstl::execution::par_unseq, constr.data.begin(),
                 constr.data.end(), body.begin(), [](tensor::Scalar x) {
                   return static_cast<tensor::DataType>(x);
                 });
  return Variable(tensor::Tensor(constr.data, constr.shape, device, dtype),
                  requires_grad);
}

}  // namespace functional

}  // namespace lmp::autograd