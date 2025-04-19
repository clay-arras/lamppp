#include "tensor.h"
#include <cassert>
#include <iostream>

namespace autograd {

Tensor Tensor::operator+(const Tensor& other) const {
  assert(shape == other.shape);
  return TensorOpFact::apply(
      [](const Tensor& a, const Tensor& b) {
        return a.as_array() + b.as_array();
      },
      this->shape, *this, other);
}

Tensor Tensor::operator-(const Tensor& other) const {
  assert(shape == other.shape);
  return TensorOpFact::apply(
      [](const Tensor& a, const Tensor& b) {
        return a.as_array() - b.as_array();
      },
      this->shape, *this, other);
}

Tensor Tensor::operator*(const Tensor& other) const {
  assert(shape == other.shape);
  return TensorOpFact::apply(
      [](const Tensor& a, const Tensor& b) {
        return a.as_array() * b.as_array();
      },
      this->shape, *this, other);
}

Tensor Tensor::operator/(const Tensor& other) const {
  assert(shape == other.shape);
  return TensorOpFact::apply(
      [](const Tensor& a, const Tensor& b) {
        return a.as_array() / b.as_array();
      },
      this->shape, *this, other);
}

Tensor Tensor::operator==(const Tensor& other) const {
  assert(shape == other.shape);
  return TensorOpFact::apply(
      [](const Tensor& a, const Tensor& b) {
        return (a.as_array() == b.as_array()).cast<float>();
      },
      this->shape, *this, other);
}

Tensor Tensor::operator!=(const Tensor& other) const {
  assert(shape == other.shape);
  return TensorOpFact::apply(
      [](const Tensor& a, const Tensor& b) {
        return (a.as_array() != b.as_array()).cast<float>();
      },
      this->shape, *this, other);
}

Tensor Tensor::operator>=(const Tensor& other) const {
  assert(shape == other.shape);
  return TensorOpFact::apply(
      [](const Tensor& a, const Tensor& b) {
        return (a.as_array() >= b.as_array()).cast<float>();
      },
      this->shape, *this, other);
}

Tensor Tensor::operator<=(const Tensor& other) const {
  assert(shape == other.shape);
  return TensorOpFact::apply(
      [](const Tensor& a, const Tensor& b) {
        return (a.as_array() <= b.as_array()).cast<float>();
      },
      this->shape, *this, other);
}

Tensor Tensor::operator>(const Tensor& other) const {
  assert(shape == other.shape);
  return TensorOpFact::apply(
      [](const Tensor& a, const Tensor& b) {
        return (a.as_array() > b.as_array()).cast<float>();
      },
      this->shape, *this, other);
}

Tensor Tensor::operator<(const Tensor& other) const {
  assert(shape == other.shape);
  return TensorOpFact::apply(
      [](const Tensor& a, const Tensor& b) {
        return (a.as_array() < b.as_array()).cast<float>();
      },
      this->shape, *this, other);
}

Tensor Tensor::matmul(const Tensor& other) const {
  assert(shape.size() == 2 && other.shape.size() == 2);
  assert(shape[1] == other.shape[0]);
  return TensorOpFact::apply(
      [](const Tensor& a, const Tensor& b) {
        // std::vector<float> res_data(a.shape[0] * b.shape[1]);
        // Eigen::Map<Eigen::MatrixXf> res(res_data.data(), a.shape[0], b.shape[1]);
        // res = (a.as_matrix(a.shape[0], a.shape[1]) *
        //         b.as_matrix(b.shape[0], b.shape[1]));
        
        // std::vector<float> ret_data(a.shape[0] * b.shape[1]);
        // Eigen::Map<Eigen::ArrayXXf> ret(ret_data.data(), a.shape[0] * b.shape[1], 1);
        // return ret;
      },
      {shape[0], other.shape[1]}, *this, other);
}

Tensor Tensor::transpose() const {
  assert(shape.size() == 2);
  return TensorOpFact::apply(
      [](const Tensor& a) {
        return a.as_matrix(a.shape[0], a.shape[1]).transpose().array();
      },
      {shape[1], shape[0]}, *this);
}

Tensor Tensor::log() const {
  return TensorOpFact::apply([](const Tensor& a) { return a.as_array().log(); },
                             this->shape, *this);
}

Tensor Tensor::exp() const {
  return TensorOpFact::apply([](const Tensor& a) { return a.as_array().exp(); },
                             this->shape, *this);
}

Tensor Tensor::relu() const {
  return TensorOpFact::apply(
      [](const Tensor& a) { return a.as_array().max(0.0F); }, this->shape,
      *this);
}

Tensor Tensor::sum(int axis) const {
  assert(axis >= 0 && axis < static_cast<int>(shape.size()));
  std::vector<int> new_shape = shape;
  new_shape[axis] = 1;

  std::vector<float> res_data(data.size() / shape[axis]);
  if (axis == 0) {
    Eigen::Map<Eigen::Matrix<float, 1, Eigen::Dynamic>> res(res_data.data(), 1,
                                                            shape[1]);
    res = as_array().reshaped(shape[0], shape[1]).colwise().sum().array();
  } else if (axis == 1) {
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>> res(res_data.data(),
                                                            shape[0], 1);
    res = as_array().reshaped(shape[0], shape[1]).rowwise().sum().array();
  } else {
    assert(
        0);  // TODO(nlin): need to implement general thingy with collapsing!!!!
  }  // TODO(nlin): this is disgusting please refactor ASAP
  return Tensor(res_data, new_shape);
}

Tensor Tensor::max(int axis) const {
  assert(axis >= 0 && axis < static_cast<int>(shape.size()));
  std::vector<int> new_shape = shape;
  new_shape[axis] = 1;

  std::vector<float> res_data(data.size() / shape[axis]);
  if (axis == 0) {
    Eigen::Map<Eigen::Matrix<float, 1, Eigen::Dynamic>> res(res_data.data(), 1,
                                                            shape[1]);
    res = as_array().reshaped(shape[0], shape[1]).colwise().maxCoeff().array();
  } else if (axis == 1) {
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>> res(res_data.data(),
                                                            shape[0], 1);
    res = as_array().reshaped(shape[0], shape[1]).rowwise().maxCoeff().array();
  } else {
    assert(0);
  }
  return Tensor(res_data, new_shape);
}

const int kMaxPrintNumel = 20;

std::ostream& operator<<(std::ostream& os, const Tensor& obj) {
  os << "Tensor(data=[";
  for (size_t i = 0; i < obj.data.size(); i++) {
    os << obj.data[i];
    if (i >= kMaxPrintNumel) {
      os << "...";
      break;
    }
    if (i < obj.data.size() - 1) {
      os << ", ";
    }
  }
  os << "], shape=[";
  for (size_t i = 0; i < obj.shape.size(); i++) {
    os << obj.shape[i];
    if (i < obj.shape.size() - 1) {
      os << ", ";
    }
  }
  os << "])";
  return os;
}

}  // namespace autograd