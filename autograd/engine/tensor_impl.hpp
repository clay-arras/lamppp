#pragma once

#ifndef TENSOR_IMPL_H
#define TENSOR_IMPL_H

#include <cassert>
#include <iostream>
#include <memory>
#include <vector>
#include "autograd/engine/backend.hpp"
#include <cstddef>
#include "autograd/engine/data_type.hpp"
#include "autograd/engine/storage.hpp"
#include "scalar.hpp"

namespace autograd {

class TensorImpl {
public:
  TensorImpl() = delete;
  ~TensorImpl() = default;

  void* data() const { return data_.data(); }
  DataType type() const { return data_.type(); };
  const std::vector<size_t>& shape() const { return data_.shape(); }
  size_t size() const { return data_.size(); }

  TensorImpl(const Storage& storage, std::shared_ptr<AbstractBackend> backend)
      : data_(storage), backend_(backend) {}

  TensorImpl add(const TensorImpl& a, const TensorImpl& b);
  TensorImpl sub(const TensorImpl& a, const TensorImpl& b);
  TensorImpl mul(const TensorImpl& a, const TensorImpl& b);
  TensorImpl div(const TensorImpl& a, const TensorImpl& b);

  TensorImpl log(const TensorImpl& a);
  TensorImpl exp(const TensorImpl& a);
  TensorImpl relu(const TensorImpl& a);

  TensorImpl matmul(const TensorImpl& a, const TensorImpl& b);
  TensorImpl transpose(const TensorImpl& a);

  TensorImpl equal(const TensorImpl& a, const TensorImpl& b);
  TensorImpl not_equal(const TensorImpl& a, const TensorImpl& b);
  TensorImpl greater_equal(const TensorImpl& a, const TensorImpl& b);
  TensorImpl less_equal(const TensorImpl& a, const TensorImpl& b);
  TensorImpl greater_than(const TensorImpl& a, const TensorImpl& b);
  TensorImpl less_than(const TensorImpl& a, const TensorImpl& b);

  TensorImpl sum(const TensorImpl& a, size_t axis);
  TensorImpl max(const TensorImpl& a, size_t axis);

  inline void fill(Scalar t) {
    DISPATCH_ALL_TYPES(type(), [&]{
      scalar_t* st = static_cast<scalar_t*>(data());
      std::fill(st, st + data_.size(), static_cast<scalar_t>(t));
    });
  }

  friend std::ostream& operator<<(std::ostream& os, const TensorImpl& obj) {
    os << "TensorImpl(data_=" << obj.data_;
    os << ", backend=" << obj.backend_;
    os << ")";
    return os;
  }
    
// private:
  std::shared_ptr<AbstractBackend> backend_; 
  Storage data_;
};

// struct TensorImpl : public virtual AbstractBackend {
//   virtual ~TensorImpl() = default;
//   virtual std::shared_ptr<TensorImpl> clone() const = 0;

//   virtual const void* data_ptr() const = 0;
//   virtual const size_t data_size() const = 0; 
//   virtual const std::vector<size_t>& shape() const = 0;

//   virtual void fill(any_type item) = 0;

//   virtual void print(std::ostream& os) const = 0;
//   friend std::ostream& operator<<(std::ostream& os, const TensorImpl& obj) {
//     obj.print(os);
//     return os;
//   }
// };

// template <typename DataType, typename Backend>
// class TensorImplModel : public TensorImpl, public virtual Backend {
//  private:
//   std::vector<DataType> _data;
//   std::vector<size_t> _shape;

//  public:
//   TensorImplModel(const std::vector<DataType>& data,
//                   const std::vector<size_t>& shape)
//       : _data(data), _shape(shape){};

//   std::shared_ptr<TensorImpl> clone() const override {
//     return std::make_shared<TensorImplModel<DataType, Backend>>(_data, _shape);
//   }

//   const void* data_ptr() const override {
//     return static_cast<const void*>(_data.data());
//   }
//   const size_t data_size() const override {
//     return static_cast<size_t>(_data.size());
//   }
//   const std::vector<size_t>& shape() const override { return _shape; }

//   void fill(any_type item) override {
//     DataType value;
//     bool converted = false;

//     std::visit(
//         [&](auto&& arg) {
//           using ArgType = std::decay_t<decltype(arg)>;
//           if constexpr (std::is_convertible_v<ArgType, DataType>) {
//             value = static_cast<DataType>(arg);
//             converted = true;
//           }
//         },
//         item);

//     assert(converted && "Unsupported type for fill operation");
//     std::fill(_data.begin(), _data.end(), value);
//   }

//   const size_t kMaxPrintNumel = 20;
//   void print(std::ostream& os) const override {
//     os << "TensorImpl(data=[";
//     for (size_t i = 0; i < data_size(); i++) {
//       os << static_cast<const DataType*>(data_ptr())[i];
//       if (i >= kMaxPrintNumel) {
//         os << "...";
//         break;
//       }
//       if (i < data_size() - 1) {
//         os << ", ";
//       }
//     }
//     os << "], shape=[";
//     for (size_t i = 0; i < shape().size(); i++) {
//       os << shape()[i];
//       if (i < shape().size() - 1) {
//         os << ", ";
//       }
//     }
//     os << "], dataType=" << typeid(DataType).name();
//     os << ", backend=" << typeid(Backend).name() << ")";
//   }
// };

}  // namespace autograd

#endif  // TENSOR_IMPL_H