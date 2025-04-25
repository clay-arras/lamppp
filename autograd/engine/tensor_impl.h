#pragma once

#ifndef TENSOR_IMPL_H
#define TENSOR_IMPL_H

#include <vector>

namespace autograd {

struct TensorImpl {
  std::vector<float> data;
  std::vector<int> shape;

  TensorImpl(const std::vector<float>& data, const std::vector<int>& shape)
      : data(data), shape(shape){};
};

}  // namespace autograd

#endif  // TENSOR_IMPL_H