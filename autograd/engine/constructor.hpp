#pragma once

#ifndef _CONSTRUCTOR_H_
#define _CONSTRUCTOR_H_

#include <cassert>
#include "variable.hpp"

namespace autograd {

inline namespace functional {

Variable zeros(const std::vector<int>& shape, bool requires_grad = false);
Variable ones(const std::vector<int>& shape, bool requires_grad = false);
Variable rand(const std::vector<int>& shape, bool requires_grad = false);

template <typename>
struct IsVector : std::false_type {};
template <typename U, typename Alloc>
struct IsVector<std::vector<U, Alloc>> : std::true_type {};

struct TensorHelper {
  std::vector<float> data;
  std::vector<int> shape;
  template <class T>
  void unroll(const std::vector<T>& tensor, int depth = 0) {
    if (depth >= shape.size()) {
      shape.push_back(tensor.size());
    }
    assert(tensor.size() == shape[depth]);
    if constexpr (IsVector<T>::value) {
      for (const T& t : tensor) {
        unroll(t, depth + 1);
      }
    } else {
      data.insert(data.end(), tensor.begin(),
                  tensor.end());  // TODO(nlin): can use memcpy, it's faster
    }
  }
};

template <class V>
Variable tensor(const std::vector<V>& data, bool requires_grad = false) {
  TensorHelper constr;
  constr.unroll(data);
  return Variable(Tensor(constr.data, constr.shape), requires_grad);
}

}  // namespace functional

#endif  // _CONSTRUCTOR_H_
}