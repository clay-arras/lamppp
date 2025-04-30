#pragma once

#ifndef TENSOR_IMPL_H
#define TENSOR_IMPL_H

#include <cassert>
#include <iostream>
#include <variant>
#include <memory>
#include <vector>
#include "supported_types.hpp"

namespace autograd {

struct TensorImpl {
  virtual ~TensorImpl() = default;
  virtual std::shared_ptr<TensorImpl> clone() const = 0;

  virtual const void* data_ptr() const = 0;
  virtual void* data_ptr() = 0;
  virtual const int data_size()
      const = 0;  // TODO: I really have to switch to size_t, this just bothers me
  virtual const std::vector<int>& shape() const = 0;

  virtual std::shared_ptr<TensorImpl> add(const TensorImpl& other) = 0;
  virtual std::shared_ptr<TensorImpl> sub(const TensorImpl& other) = 0;
  virtual std::shared_ptr<TensorImpl> mul(const TensorImpl& other) = 0;
  virtual std::shared_ptr<TensorImpl> div(const TensorImpl& other) = 0;

  virtual std::shared_ptr<TensorImpl> log() = 0;
  virtual std::shared_ptr<TensorImpl> exp() = 0;
  virtual std::shared_ptr<TensorImpl> relu() = 0;

  virtual std::shared_ptr<TensorImpl> matmul(const TensorImpl& other) = 0;
  virtual std::shared_ptr<TensorImpl> transpose() = 0;

  virtual std::shared_ptr<TensorImpl> equal(const TensorImpl& other) = 0;
  virtual std::shared_ptr<TensorImpl> not_equal(const TensorImpl& other) = 0;
  virtual std::shared_ptr<TensorImpl> greater_equal(const TensorImpl& other) = 0;
  virtual std::shared_ptr<TensorImpl> less_equal(const TensorImpl& other) = 0;
  virtual std::shared_ptr<TensorImpl> greater_than(const TensorImpl& other) = 0;
  virtual std::shared_ptr<TensorImpl> less_than(const TensorImpl& other) = 0;

  virtual std::shared_ptr<TensorImpl> sum(int axis) = 0;
  virtual std::shared_ptr<TensorImpl> max(int axis) = 0;

  virtual void fill(any_type item) = 0;

  virtual void print(std::ostream& os) const = 0;
  friend std::ostream& operator<<(std::ostream& os, const TensorImpl& obj) {
    obj.print(os);
    return os;
  }
};

template <typename DataType, typename Backend>
class TensorImplModel : public TensorImpl {
 public:
  TensorImplModel(const std::vector<DataType>& data,
                  const std::vector<int>& shape)
      : _data(data), _shape(shape){};

  std::shared_ptr<TensorImpl> clone() const override {
      return std::make_shared<TensorImplModel<DataType, Backend>>(_data, _shape);
  }

  const void* data_ptr() const override {
    return static_cast<const void*>(_data.data());
  }
  void* data_ptr() override {
    return static_cast<void*>(_data.data());
  }
  const int data_size() const override {
    return static_cast<int>(_data.size());
  }
  const std::vector<int>& shape() const override { return _shape; }

  std::shared_ptr<TensorImpl> add(const TensorImpl& other) override {
    return Backend().add(*this, other);
  }
  std::shared_ptr<TensorImpl> sub(const TensorImpl& other) override {
    return Backend().sub(*this, other);
  }
  std::shared_ptr<TensorImpl> mul(const TensorImpl& other) override {
    return Backend().mul(*this, other);
  }
  std::shared_ptr<TensorImpl> div(const TensorImpl& other) override {
    return Backend().div(*this, other);
  }

  std::shared_ptr<TensorImpl> log() override {
    return Backend().log(*this);
  }
  std::shared_ptr<TensorImpl> exp() override {
    return Backend().exp(*this);
  }
  std::shared_ptr<TensorImpl> relu() override {
    return Backend().relu(*this);
  }

  std::shared_ptr<TensorImpl> matmul(const TensorImpl& other) override {
    return Backend().matmul(*this, other);
  }
  std::shared_ptr<TensorImpl> transpose() override {
    return Backend().transpose(*this);
  }

  std::shared_ptr<TensorImpl> equal(const TensorImpl& other) override {
    return Backend().equal(*this, other);
  }
  std::shared_ptr<TensorImpl> not_equal(const TensorImpl& other) override {
    return Backend().not_equal(*this, other);
  }
  std::shared_ptr<TensorImpl> greater_equal(const TensorImpl& other) override {
    return Backend().greater_equal(*this, other);
  }
  std::shared_ptr<TensorImpl> less_equal(const TensorImpl& other) override {
    return Backend().less_equal(*this, other);
  }
  std::shared_ptr<TensorImpl> greater_than(const TensorImpl& other) override {
    return Backend().greater_than(*this, other);
  }
  std::shared_ptr<TensorImpl> less_than(const TensorImpl& other) override {
    return Backend().less_than(*this, other);
  }

  std::shared_ptr<TensorImpl> sum(int axis) override {
    return Backend().sum(*this, axis);
  }
  std::shared_ptr<TensorImpl> max(int axis) override {
    return Backend().max(*this, axis);
  }

  void fill(any_type item) override {
    if (std::holds_alternative<int>(item)) {
        int i = std::get<int>(item);
        DataType fillItem = static_cast<DataType>(i);
        std::fill(_data.begin(), _data.end(), fillItem);
    } else if (std::holds_alternative<float>(item)) {
        float f = std::get<float>(item);
        DataType fillItem = static_cast<DataType>(f);
        std::fill(_data.begin(), _data.end(), fillItem);
    } else if (std::holds_alternative<double>(item)) {
        double d = std::get<double>(item);
        DataType fillItem = static_cast<DataType>(d);
        std::fill(_data.begin(), _data.end(), fillItem);
    } else {
        assert(false && "Unsupported type for fill operation");
    }
  }

  const int kMaxPrintNumel = 20;
  void print(std::ostream& os) const override {
    os << "TensorImpl(data=[";
    for (size_t i = 0; i < data_size(); i++) {
      os << static_cast<const DataType*>(data_ptr())[i];
      if (i >= kMaxPrintNumel) {
        os << "...";
        break;
      }
      if (i < data_size() - 1) {
        os << ", ";
      }
    }
    os << "], shape=[";
    for (size_t i = 0; i < shape().size(); i++) {
      os << shape()[i];
      if (i < shape().size() - 1) {
        os << ", ";
      }
    }
    os << "], dataType=" << typeid(DataType).name();
    os << ", backend=" << typeid(Backend).name() << ")";
  }

 private:
  std::vector<DataType> _data;
  std::vector<int> _shape;
};

}  // namespace autograd

#endif  // TENSOR_IMPL_H