#pragma once

#include <cassert>
#ifndef TENSOR_IMPL_H
#define TENSOR_IMPL_H

#include <memory>
#include <vector>

namespace autograd {

struct TensorImpl {
  virtual ~TensorImpl() = default;
  virtual std::shared_ptr<TensorImpl> clone() const = 0;

  virtual const void* data_ptr() const = 0;
  virtual const int data_size()
      const = 0;  // TODO: I really have to switch to size_t, this just bothers me
  virtual const std::vector<int>& shape() const = 0;

  virtual std::shared_ptr<TensorImpl> add(const TensorImpl& a, // TODO: these can just take one argument, other
                                          const TensorImpl& b) = 0;
  virtual std::shared_ptr<TensorImpl> sub(const TensorImpl& a,
                                          const TensorImpl& b) = 0;
  virtual std::shared_ptr<TensorImpl> mul(const TensorImpl& a,
                                          const TensorImpl& b) = 0;
  virtual std::shared_ptr<TensorImpl> div(const TensorImpl& a,
                                          const TensorImpl& b) = 0;

  virtual std::shared_ptr<TensorImpl> log(const TensorImpl& a) = 0;
  virtual std::shared_ptr<TensorImpl> exp(const TensorImpl& a) = 0;
  virtual std::shared_ptr<TensorImpl> relu(const TensorImpl& a) = 0;

  virtual std::shared_ptr<TensorImpl> matmul(const TensorImpl& a,
                                             const TensorImpl& b) = 0;
  virtual std::shared_ptr<TensorImpl> transpose(const TensorImpl& a) = 0;

  virtual std::shared_ptr<TensorImpl> equal(const TensorImpl& a,
                                            const TensorImpl& b) = 0;
  virtual std::shared_ptr<TensorImpl> not_equal(const TensorImpl& a,
                                                const TensorImpl& b) = 0;
  virtual std::shared_ptr<TensorImpl> greater_equal(const TensorImpl& a,
                                                    const TensorImpl& b) = 0;
  virtual std::shared_ptr<TensorImpl> less_equal(const TensorImpl& a,
                                                 const TensorImpl& b) = 0;
  virtual std::shared_ptr<TensorImpl> greater_than(const TensorImpl& a,
                                                   const TensorImpl& b) = 0;
  virtual std::shared_ptr<TensorImpl> less_than(const TensorImpl& a,
                                                const TensorImpl& b) = 0;

  virtual std::shared_ptr<TensorImpl> sum(const TensorImpl& a, int axis) = 0;
  virtual std::shared_ptr<TensorImpl> max(const TensorImpl& a, int axis) = 0;

  virtual void fill(void* item) = 0;

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
  const int data_size() const override {
    return static_cast<int>(_data.size());
  }
  const std::vector<int>& shape() const override { return _shape; }

  std::shared_ptr<TensorImpl> add(const TensorImpl& a,
                                  const TensorImpl& b) override {
    return Backend().add(a, b);
  }
  std::shared_ptr<TensorImpl> sub(const TensorImpl& a,
                                  const TensorImpl& b) override {
    return Backend().sub(a, b);
  }
  std::shared_ptr<TensorImpl> mul(const TensorImpl& a,
                                  const TensorImpl& b) override {
    return Backend().mul(a, b);
  }
  std::shared_ptr<TensorImpl> div(const TensorImpl& a,
                                  const TensorImpl& b) override {
    return Backend().div(a, b);
  }

  std::shared_ptr<TensorImpl> log(const TensorImpl& a) override {
    return Backend().log(a);
  }
  std::shared_ptr<TensorImpl> exp(const TensorImpl& a) override {
    return Backend().exp(a);
  }
  std::shared_ptr<TensorImpl> relu(const TensorImpl& a) override {
    return Backend().relu(a);
  }

  std::shared_ptr<TensorImpl> matmul(const TensorImpl& a,
                                     const TensorImpl& b) override {
    return Backend().matmul(a, b);
  }
  std::shared_ptr<TensorImpl> transpose(const TensorImpl& a) override {
    return Backend().transpose(a);
  }

  std::shared_ptr<TensorImpl> equal(const TensorImpl& a,
                                    const TensorImpl& b) override {
    return Backend().equal(a, b);
  }
  std::shared_ptr<TensorImpl> not_equal(const TensorImpl& a,
                                        const TensorImpl& b) override {
    return Backend().not_equal(a, b);
  }
  std::shared_ptr<TensorImpl> greater_equal(const TensorImpl& a,
                                            const TensorImpl& b) override {
    return Backend().greater_equal(a, b);
  }
  std::shared_ptr<TensorImpl> less_equal(const TensorImpl& a,
                                         const TensorImpl& b) override {
    return Backend().less_equal(a, b);
  }
  std::shared_ptr<TensorImpl> greater_than(const TensorImpl& a,
                                           const TensorImpl& b) override {
    return Backend().greater_than(a, b);
  }
  std::shared_ptr<TensorImpl> less_than(const TensorImpl& a,
                                        const TensorImpl& b) override {
    return Backend().less_than(a, b);
  }

  std::shared_ptr<TensorImpl> sum(const TensorImpl& a, int axis) override {
    return Backend().sum(a, axis);
  }
  std::shared_ptr<TensorImpl> max(const TensorImpl& a, int axis) override {
    return Backend().max(a, axis);
  }

  void fill(void* item) override {
    if (auto ptr = static_cast<DataType *>(item)) {
      std::fill(_data.begin(), _data.end(), *ptr);
    } else {
      assert(false);
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