#pragma once

#ifndef _CONSTRUCTOR_H_
#define _CONSTRUCTOR_H_

#include <Eigen/Core>
#include <cassert>
#include <numeric>
#include "autograd/engine/scalar.hpp"
#include "variable.hpp"

namespace autograd {

inline namespace functional {

using std::multiplies;

template <typename DataType, typename Backend>
Variable zeros(const std::vector<size_t>& shape, bool requires_grad) {
  size_t sz = std::accumulate(shape.begin(), shape.end(), 1, multiplies<>());
  return Variable(
      Tensor::create<DataType, Backend>(std::vector<DataType>(sz, 0.0), shape),
      requires_grad);
}

template <typename DataType, typename Backend>
Variable ones(const std::vector<size_t>& shape, bool requires_grad) {
  size_t sz = std::accumulate(shape.begin(), shape.end(), 1, multiplies<>());
  return Variable(
      Tensor::create<DataType, Backend>(std::vector<DataType>(sz, 1.0), shape),
      requires_grad);
}

template <typename DataType, typename Backend>
Variable rand(const std::vector<size_t>& shape, bool requires_grad) {
  // TODO: FIX THIS!!!; this is broken because ArrayXXf only supports floats
  size_t sz = std::accumulate(shape.begin(), shape.end(), 1, multiplies<>());
  std::vector<DataType> rand_vec(sz);
  Eigen::Map<Eigen::ArrayXXf> res(rand_vec.data(), sz, 1);
  res.setRandom();
  return Variable(Tensor::create<DataType, Backend>(rand_vec, shape),
                  requires_grad);
}

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

template <typename V, typename DataType, typename Backend>
Variable tensor(const std::vector<V>& data, bool requires_grad = false) {
  TensorHelper constr;
  constr.unroll(data);
  std::vector<DataType> body(constr.data.size());
  std::transform(__pstl::execution::par_unseq, constr.data.begin(),
                 constr.data.end(), body.begin(),
                 [](Scalar x) { return static_cast<DataType>(x); });
  return Variable(Tensor::create<DataType, Backend>(constr.data, constr.shape),
                  requires_grad);
}

}  // namespace functional

#endif  // _CONSTRUCTOR_H_
}