#include "include/lamppp/tensor/functions/binary_ops.hpp"
#include "include/lamppp/tensor/align_utils.hpp"
#include "include/lamppp/tensor/cuda/binary_kern.cuh"
#include "include/lamppp/tensor/data_type.hpp"
#include "include/lamppp/tensor/tensor.hpp"
#include "include/lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::ops {

LMP_DEFINE_DISPATCH(equal_stub);
LMP_DEFINE_DISPATCH(not_equal_stub);
LMP_DEFINE_DISPATCH(greater_equal_stub);
LMP_DEFINE_DISPATCH(less_equal_stub);
LMP_DEFINE_DISPATCH(greater_stub);
LMP_DEFINE_DISPATCH(less_stub);

TensorImpl equal_cpu(const TensorImpl& a, const TensorImpl& b) {
  assert(false && "Not Implemented");
}

TensorImpl equal_cuda(const TensorImpl& a, const TensorImpl& b) {
  detail::AlignUtil meta(a.shape(), b.shape());
  detail::cuda::OffsetUtil offset(a.shape(), b.shape(), a.strides(),
                                  b.strides(), meta.aligned_stride_,
                                  meta.aligned_shape_.size());
  DataType out_dtype = DataType::Bool;
  Storage c_storage(meta.aligned_size_ * sizeof(bool), DeviceType::CUDA);
  LMP_DISPATCH_ALL_TYPES(a.type(), [&] {
    using a_type_t = scalar_t;
    LMP_DISPATCH_ALL_TYPES(b.type(), [&] {
      using b_type_t = scalar_t;
      ::lmp::tensor::detail::cuda::vecEqual<a_type_t, b_type_t>(
          meta.aligned_size_, static_cast<const a_type_t*>(a.data()),
          static_cast<const b_type_t*>(b.data()),
          static_cast<bool*>(c_storage.data()), &offset);
    });
  });
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  assert(err == cudaSuccess && "equal_cuda: CUDA error after synchronize.");
  return TensorImpl(c_storage, meta.aligned_shape_, out_dtype);
}

TensorImpl not_equal_cpu(const TensorImpl& a, const TensorImpl& b) {
  assert(false && "Not Implemented");
}

TensorImpl not_equal_cuda(const TensorImpl& a, const TensorImpl& b) {
  detail::AlignUtil meta(a.shape(), b.shape());
  detail::cuda::OffsetUtil offset(a.shape(), b.shape(), a.strides(),
                                  b.strides(), meta.aligned_stride_,
                                  meta.aligned_shape_.size());
  DataType out_dtype = DataType::Bool;
  Storage c_storage(meta.aligned_size_ * sizeof(bool), DeviceType::CUDA);
  LMP_DISPATCH_ALL_TYPES(a.type(), [&] {
    using a_type_t = scalar_t;
    LMP_DISPATCH_ALL_TYPES(b.type(), [&] {
      using b_type_t = scalar_t;
      ::lmp::tensor::detail::cuda::vecNotEqual<a_type_t, b_type_t>(
          meta.aligned_size_, static_cast<const a_type_t*>(a.data()),
          static_cast<const b_type_t*>(b.data()),
          static_cast<bool*>(c_storage.data()), &offset);
    });
  });
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  assert(err == cudaSuccess && "not_equal_cuda: CUDA error after synchronize.");
  return TensorImpl(c_storage, meta.aligned_shape_, out_dtype);
}

TensorImpl greater_equal_cpu(const TensorImpl& a, const TensorImpl& b) {
  assert(false && "Not Implemented");
}

TensorImpl greater_equal_cuda(const TensorImpl& a, const TensorImpl& b) {
  detail::AlignUtil meta(a.shape(), b.shape());
  detail::cuda::OffsetUtil offset(a.shape(), b.shape(), a.strides(),
                                  b.strides(), meta.aligned_stride_,
                                  meta.aligned_shape_.size());
  DataType out_dtype = DataType::Bool;
  Storage c_storage(meta.aligned_size_ * sizeof(bool), DeviceType::CUDA);
  LMP_DISPATCH_ALL_TYPES(a.type(), [&] {
    using a_type_t = scalar_t;
    LMP_DISPATCH_ALL_TYPES(b.type(), [&] {
      using b_type_t = scalar_t;
      ::lmp::tensor::detail::cuda::vecGreaterEqual<a_type_t, b_type_t>(
          meta.aligned_size_, static_cast<const a_type_t*>(a.data()),
          static_cast<const b_type_t*>(b.data()),
          static_cast<bool*>(c_storage.data()), &offset);
    });
  });
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  assert(err == cudaSuccess &&
         "greater_equal_cuda: CUDA error after synchronize.");
  return TensorImpl(c_storage, meta.aligned_shape_, out_dtype);
}

TensorImpl less_equal_cpu(const TensorImpl& a, const TensorImpl& b) {
  assert(false && "Not Implemented");
}

TensorImpl less_equal_cuda(const TensorImpl& a, const TensorImpl& b) {
  detail::AlignUtil meta(a.shape(), b.shape());
  detail::cuda::OffsetUtil offset(a.shape(), b.shape(), a.strides(),
                                  b.strides(), meta.aligned_stride_,
                                  meta.aligned_shape_.size());
  DataType out_dtype = DataType::Bool;
  Storage c_storage(meta.aligned_size_ * sizeof(bool), DeviceType::CUDA);
  LMP_DISPATCH_ALL_TYPES(a.type(), [&] {
    using a_type_t = scalar_t;
    LMP_DISPATCH_ALL_TYPES(b.type(), [&] {
      using b_type_t = scalar_t;
      ::lmp::tensor::detail::cuda::vecLessEqual<a_type_t, b_type_t>(
          meta.aligned_size_, static_cast<const a_type_t*>(a.data()),
          static_cast<const b_type_t*>(b.data()),
          static_cast<bool*>(c_storage.data()), &offset);
    });
  });
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  assert(err == cudaSuccess &&
         "less_equal_cuda: CUDA error after synchronize.");
  return TensorImpl(c_storage, meta.aligned_shape_, out_dtype);
}

TensorImpl greater_cpu(const TensorImpl& a, const TensorImpl& b) {
  assert(false && "Not Implemented");
}

TensorImpl greater_cuda(const TensorImpl& a, const TensorImpl& b) {
  detail::AlignUtil meta(a.shape(), b.shape());
  detail::cuda::OffsetUtil offset(a.shape(), b.shape(), a.strides(),
                                  b.strides(), meta.aligned_stride_,
                                  meta.aligned_shape_.size());
  DataType out_dtype = DataType::Bool;
  Storage c_storage(meta.aligned_size_ * sizeof(bool), DeviceType::CUDA);
  LMP_DISPATCH_ALL_TYPES(a.type(), [&] {
    using a_type_t = scalar_t;
    LMP_DISPATCH_ALL_TYPES(b.type(), [&] {
      using b_type_t = scalar_t;
      ::lmp::tensor::detail::cuda::vecGreaterThan<a_type_t, b_type_t>(
          meta.aligned_size_, static_cast<const a_type_t*>(a.data()),
          static_cast<const b_type_t*>(b.data()),
          static_cast<bool*>(c_storage.data()), &offset);
    });
  });
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  assert(err == cudaSuccess && "greater_cuda: CUDA error after synchronize.");
  return TensorImpl(c_storage, meta.aligned_shape_, out_dtype);
}

TensorImpl less_cpu(const TensorImpl& a, const TensorImpl& b) {
  assert(false && "Not Implemented");
}

TensorImpl less_cuda(const TensorImpl& a, const TensorImpl& b) {
  detail::AlignUtil meta(a.shape(), b.shape());
  detail::cuda::OffsetUtil offset(a.shape(), b.shape(), a.strides(),
                                  b.strides(), meta.aligned_stride_,
                                  meta.aligned_shape_.size());
  DataType out_dtype = DataType::Bool;
  Storage c_storage(meta.aligned_size_ * sizeof(bool), DeviceType::CUDA);
  LMP_DISPATCH_ALL_TYPES(a.type(), [&] {
    using a_type_t = scalar_t;
    LMP_DISPATCH_ALL_TYPES(b.type(), [&] {
      using b_type_t = scalar_t;
      ::lmp::tensor::detail::cuda::vecLessThan<a_type_t, b_type_t>(
          meta.aligned_size_, static_cast<const a_type_t*>(a.data()),
          static_cast<const b_type_t*>(b.data()),
          static_cast<bool*>(c_storage.data()), &offset);
    });
  });
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  assert(err == cudaSuccess && "less_cuda: CUDA error after synchronize.");
  return TensorImpl(c_storage, meta.aligned_shape_, out_dtype);
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

}  // namespace lmp::tensor::ops