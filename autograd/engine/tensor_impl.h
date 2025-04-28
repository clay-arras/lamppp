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

// struct TensorImpl {
//   virtual ~TensorImpl() = default;
//   virtual void* data_ptr() = 0;
//   virtual int data_size() = 0; // TODO: I really have to switch to size_t, this just bothers me
//   virtual std::vector<int> shape() = 0;
// };

// template <typename DataType>
// class TensorImplModel : TensorImpl {
// public:
//   using type = DataType;
//   TensorImplModel(const std::vector<DataType>& data, const std::vector<int>& shape)
//       : _data(data), _shape(shape){};

// private:
//   std::vector<DataType> _data;
//   std::vector<int> _shape;
// };

}  // namespace autograd

#endif  // TENSOR_IMPL_H