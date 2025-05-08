#pragma once

#include <cassert>
#include "autograd/engine/scalar.hpp"
#include "variable.hpp"

namespace autograd {

inline namespace functional {

using std::multiplies;

Variable zeros(const std::vector<size_t>& shape, DeviceType device,
               DataType dtype, bool requires_grad);

Variable ones(const std::vector<size_t>& shape, DeviceType device,
              DataType dtype, bool requires_grad);

Variable rand(const std::vector<size_t>& shape, DeviceType device,
              DataType dtype, bool requires_grad);

template <typename>
struct IsVector : std::false_type {};
template <typename U, typename Alloc>
struct IsVector<std::vector<U, Alloc>> : std::true_type {};

struct TensorHelper {
  std::vector<Scalar> data;
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
Variable tensor(const std::vector<V>& data, DeviceType device, DataType dtype,
                bool requires_grad) {
  TensorHelper constr;
  constr.unroll(data);
  std::vector<DataType> body(constr.data.size());
  std::transform(__pstl::execution::par_unseq, constr.data.begin(),
                 constr.data.end(), body.begin(),
                 [](Scalar x) { return static_cast<DataType>(x); });
  return Variable(Tensor(constr.data, constr.shape, device, dtype),
                  requires_grad);
}

}  // namespace functional

}  // namespace autograd