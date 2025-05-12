#include "include/lamppp/tensor/functions/binary_ops.hpp"
#include "include/lamppp/tensor/cuda/binary_kern.cuh"
#include "include/lamppp/tensor/data_type.hpp"
#include "include/lamppp/tensor/tensor.hpp"
#include "include/lamppp/tensor/tensor_impl.hpp"

namespace autograd {

DEFINE_DISPATCH(equal_stub);
DEFINE_DISPATCH(not_equal_stub);
DEFINE_DISPATCH(greater_equal_stub);
DEFINE_DISPATCH(less_equal_stub);
DEFINE_DISPATCH(greater_stub);
DEFINE_DISPATCH(less_stub);

TensorImpl equal_cpu(const TensorImpl& a, const TensorImpl& b) {
  assert(false && "Not Implemented");
}

TensorImpl equal_cuda(const TensorImpl& a, const TensorImpl& b) {
  assert(a.size() == b.size() && "Size mismatch");
  assert(a.shape() == b.shape() && "Shape mismatch");

  DataType out_dtype = DataType::Bool;
  Storage c(a.size() * sizeof(bool), DeviceType::CUDA);
  DISPATCH_ALL_TYPES(a.type(), [&] {
    using a_type_t = scalar_t;
    DISPATCH_ALL_TYPES(b.type(), [&] {
      using b_type_t = scalar_t;
      vecEqual<a_type_t, b_type_t>(
          a.size(), static_cast<const a_type_t*>(a.data()),
          static_cast<const b_type_t*>(b.data()), static_cast<bool*>(c.data()));
    });
  });
  return TensorImpl(c, a.shape(), out_dtype);
}

TensorImpl not_equal_cpu(const TensorImpl& a, const TensorImpl& b) {
  assert(false && "Not Implemented");
}

TensorImpl not_equal_cuda(const TensorImpl& a, const TensorImpl& b) {
  assert(a.size() == b.size() && "Size mismatch");
  assert(a.shape() == b.shape() && "Shape mismatch");

  DataType out_dtype = DataType::Bool;
  Storage c(a.size() * sizeof(bool), DeviceType::CUDA);
  DISPATCH_ALL_TYPES(a.type(), [&] {
    using a_type_t = scalar_t;
    DISPATCH_ALL_TYPES(b.type(), [&] {
      using b_type_t = scalar_t;
      vecNotEqual<a_type_t, b_type_t>(
          a.size(), static_cast<const a_type_t*>(a.data()),
          static_cast<const b_type_t*>(b.data()), static_cast<bool*>(c.data()));
    });
  });
  return TensorImpl(c, a.shape(), out_dtype);
}

TensorImpl greater_equal_cpu(const TensorImpl& a, const TensorImpl& b) {
  assert(false && "Not Implemented");
}

TensorImpl greater_equal_cuda(const TensorImpl& a, const TensorImpl& b) {
  assert(a.size() == b.size() && "Size mismatch");
  assert(a.shape() == b.shape() && "Shape mismatch");

  DataType out_dtype = DataType::Bool;
  Storage c(a.size() * sizeof(bool), DeviceType::CUDA);
  DISPATCH_ALL_TYPES(a.type(), [&] {
    using a_type_t = scalar_t;
    DISPATCH_ALL_TYPES(b.type(), [&] {
      using b_type_t = scalar_t;
      vecGreaterEqual<a_type_t, b_type_t>(
          a.size(), static_cast<const a_type_t*>(a.data()),
          static_cast<const b_type_t*>(b.data()), static_cast<bool*>(c.data()));
    });
  });
  return TensorImpl(c, a.shape(), out_dtype);
}

TensorImpl less_equal_cpu(const TensorImpl& a, const TensorImpl& b) {
  assert(false && "Not Implemented");
}

TensorImpl less_equal_cuda(const TensorImpl& a, const TensorImpl& b) {
  assert(a.size() == b.size() && "Size mismatch");
  assert(a.shape() == b.shape() && "Shape mismatch");

  DataType out_dtype = DataType::Bool;
  Storage c(a.size() * sizeof(bool), DeviceType::CUDA);
  DISPATCH_ALL_TYPES(a.type(), [&] {
    using a_type_t = scalar_t;
    DISPATCH_ALL_TYPES(b.type(), [&] {
      using b_type_t = scalar_t;
      vecLessEqual<a_type_t, b_type_t>(
          a.size(), static_cast<const a_type_t*>(a.data()),
          static_cast<const b_type_t*>(b.data()), static_cast<bool*>(c.data()));
    });
  });
  return TensorImpl(c, a.shape(), out_dtype);
}

TensorImpl greater_cpu(const TensorImpl& a, const TensorImpl& b) {
  assert(false && "Not Implemented");
}

TensorImpl greater_cuda(const TensorImpl& a, const TensorImpl& b) {
  assert(a.size() == b.size() && "Size mismatch");
  assert(a.shape() == b.shape() && "Shape mismatch");

  DataType out_dtype = DataType::Bool;
  Storage c(a.size() * sizeof(bool), DeviceType::CUDA);
  DISPATCH_ALL_TYPES(a.type(), [&] {
    using a_type_t = scalar_t;
    DISPATCH_ALL_TYPES(b.type(), [&] {
      using b_type_t = scalar_t;
      vecGreaterThan<a_type_t, b_type_t>(
          a.size(), static_cast<const a_type_t*>(a.data()),
          static_cast<const b_type_t*>(b.data()), static_cast<bool*>(c.data()));
    });
  });
  return TensorImpl(c, a.shape(), out_dtype);
}

TensorImpl less_cpu(const TensorImpl& a, const TensorImpl& b) {
  assert(false && "Not Implemented");
}

TensorImpl less_cuda(const TensorImpl& a, const TensorImpl& b) {
  assert(a.size() == b.size() && "Size mismatch");
  assert(a.shape() == b.shape() && "Shape mismatch");

  DataType out_dtype = DataType::Bool;
  Storage c(a.size() * sizeof(bool), DeviceType::CUDA);
  DISPATCH_ALL_TYPES(a.type(), [&] {
    using a_type_t = scalar_t;
    DISPATCH_ALL_TYPES(b.type(), [&] {
      using b_type_t = scalar_t;
      vecLessThan<a_type_t, b_type_t>(
          a.size(), static_cast<const a_type_t*>(a.data()),
          static_cast<const b_type_t*>(b.data()), static_cast<bool*>(c.data()));
    });
  });
  return TensorImpl(c, a.shape(), out_dtype);
}

Tensor equal(const Tensor& a, const Tensor& b) {
  assert(a.device() == b.device() && "Tensors are on different devices");
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      equal_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a),
                 *detail::UnsafeTensorAccessor::getImpl(b))));
}

Tensor not_equal(const Tensor& a, const Tensor& b) {
  assert(a.device() == b.device() && "Tensors are on different devices");
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      not_equal_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a),
                     *detail::UnsafeTensorAccessor::getImpl(b))));
}

Tensor greater_equal(const Tensor& a, const Tensor& b) {
  assert(a.device() == b.device() && "Tensors are on different devices");
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      greater_equal_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a),
                         *detail::UnsafeTensorAccessor::getImpl(b))));
}

Tensor less_equal(const Tensor& a, const Tensor& b) {
  assert(a.device() == b.device() && "Tensors are on different devices");
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      less_equal_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a),
                      *detail::UnsafeTensorAccessor::getImpl(b))));
}

Tensor greater(const Tensor& a, const Tensor& b) {
  assert(a.device() == b.device() && "Tensors are on different devices");
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      greater_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a),
                   *detail::UnsafeTensorAccessor::getImpl(b))));
}

Tensor less(const Tensor& a, const Tensor& b) {
  assert(a.device() == b.device() && "Tensors are on different devices");
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      less_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a),
                *detail::UnsafeTensorAccessor::getImpl(b))));
}

}  // namespace autograd