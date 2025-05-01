#pragma once

#ifndef TENSOR_IMPL_H
#define TENSOR_IMPL_H

#include "autograd/engine/backend.hpp"
#include <cassert>
#include <iostream>
#include <variant>
#include <memory>
#include <vector>
#include "supported_types.hpp"

namespace autograd {

struct TensorImpl : public virtual AbstractBackend {
  virtual ~TensorImpl() = default;
  virtual std::shared_ptr<TensorImpl> clone() const = 0;

  virtual const void* data_ptr() const = 0;
  virtual void* data_ptr() = 0;
  virtual const int data_size()
      const = 0;  // TODO: I really have to switch to size_t, this just bothers me
  virtual const std::vector<int>& shape() const = 0;

  virtual void fill(any_type item) = 0;

  virtual void print(std::ostream& os) const = 0;
  friend std::ostream& operator<<(std::ostream& os, const TensorImpl& obj) {
    obj.print(os);
    return os;
  }
};

template <typename DataType, typename Backend>
class TensorImplModel : public TensorImpl, public virtual Backend {
 private:
  std::vector<DataType> _data;
  std::vector<int> _shape;

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

  void fill(any_type item) override {
    DataType value;
    bool converted = false;
    
    std::visit([&](auto&& arg) {
      using ArgType = std::decay_t<decltype(arg)>;
      if constexpr (std::is_convertible_v<ArgType, DataType>) {
        value = static_cast<DataType>(arg);
        converted = true;
      }
    }, item);
    
    assert(converted && "Unsupported type for fill operation");
    std::fill(_data.begin(), _data.end(), value);
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
};

}  // namespace autograd

#endif  // TENSOR_IMPL_H