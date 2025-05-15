#include "include/lamppp/tensor/functions/basic_ops.hpp"
#include "include/lamppp/tensor/align_utils.hpp"
#include "include/lamppp/tensor/cuda/basic_kern.cuh"
#include "include/lamppp/tensor/cuda/offset_util.cuh"
#include "include/lamppp/tensor/data_type.hpp"
#include "include/lamppp/tensor/tensor.hpp"

namespace lmp::tensor::ops {

LMP_DEFINE_DISPATCH(add_stub);
LMP_DEFINE_DISPATCH(sub_stub);
LMP_DEFINE_DISPATCH(mul_stub);
LMP_DEFINE_DISPATCH(div_stub);

TensorImpl add_cpu(const TensorImpl& a, const TensorImpl& b) {
  assert(false && "Not Implemented");
}

TensorImpl add_cuda(const TensorImpl& a, const TensorImpl& b) {
  // NOTE: this is absolutely horrible
  detail::AlignUtil meta(a.shape(), b.shape());
  detail::cuda::OffsetUtil offset(a.shape(), b.shape(), a.strides(),
                                  b.strides(), meta.aligned_stride_,
                                  meta.aligned_shape_.size());
  DataType out_dtype = type_upcast(a.type(), b.type());
  return LMP_DISPATCH_ALL_TYPES(a.type(), [&] {
    using a_type_t = scalar_t;
    return LMP_DISPATCH_ALL_TYPES(b.type(), [&] {
      using b_type_t = scalar_t;
      return LMP_DISPATCH_ALL_TYPES(out_dtype, [&] {
        using out_type = scalar_t;
        Storage c_storage(meta.aligned_size_ * sizeof(out_type),
                          DeviceType::CUDA);

        ::lmp::tensor::detail::cuda::vecAdd<a_type_t, b_type_t, out_type>(
            meta.aligned_size_, static_cast<const a_type_t*>(a.data()),
            static_cast<const b_type_t*>(b.data()),
            static_cast<out_type*>(c_storage.data()), &offset);

        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        assert(err == cudaSuccess && "add_cuda: CUDA error after synchronize.");

        return TensorImpl(c_storage, meta.aligned_shape_, out_dtype);
      });
    });
  });
}

TensorImpl sub_cpu(const TensorImpl& a, const TensorImpl& b) {
  assert(false && "Not Implemented");
}

TensorImpl sub_cuda(const TensorImpl& a, const TensorImpl& b) {
  detail::AlignUtil meta(a.shape(), b.shape());
  detail::cuda::OffsetUtil offset(a.shape(), b.shape(), a.strides(),
                                  b.strides(), meta.aligned_stride_,
                                  meta.aligned_shape_.size());
  DataType out_dtype = type_upcast(a.type(), b.type());
  return LMP_DISPATCH_ALL_TYPES(a.type(), [&] {
    using a_type_t = scalar_t;
    return LMP_DISPATCH_ALL_TYPES(b.type(), [&] {
      using b_type_t = scalar_t;
      return LMP_DISPATCH_ALL_TYPES(out_dtype, [&] {
        using out_type = scalar_t;
        Storage c_storage(meta.aligned_size_ * sizeof(out_type),
                          DeviceType::CUDA);

        ::lmp::tensor::detail::cuda::vecSub<a_type_t, b_type_t, out_type>(
            meta.aligned_size_, static_cast<const a_type_t*>(a.data()),
            static_cast<const b_type_t*>(b.data()),
            static_cast<out_type*>(c_storage.data()), &offset);

        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        assert(err == cudaSuccess && "sub_cuda: CUDA error after synchronize.");

        return TensorImpl(c_storage, meta.aligned_shape_, out_dtype);
      });
    });
  });
}

TensorImpl mul_cpu(const TensorImpl& a, const TensorImpl& b) {
  assert(false && "Not Implemented");
}

TensorImpl mul_cuda(const TensorImpl& a, const TensorImpl& b) {
  detail::AlignUtil meta(a.shape(), b.shape());
  detail::cuda::OffsetUtil offset(a.shape(), b.shape(), a.strides(),
                                  b.strides(), meta.aligned_stride_,
                                  meta.aligned_shape_.size());
  DataType out_dtype = type_upcast(a.type(), b.type());
  return LMP_DISPATCH_ALL_TYPES(a.type(), [&] {
    using a_type_t = scalar_t;
    return LMP_DISPATCH_ALL_TYPES(b.type(), [&] {
      using b_type_t = scalar_t;
      return LMP_DISPATCH_ALL_TYPES(out_dtype, [&] {
        using out_type = scalar_t;
        Storage c_storage(meta.aligned_size_ * sizeof(out_type),
                          DeviceType::CUDA);

        ::lmp::tensor::detail::cuda::vecMul<a_type_t, b_type_t, out_type>(
            meta.aligned_size_, static_cast<const a_type_t*>(a.data()),
            static_cast<const b_type_t*>(b.data()),
            static_cast<out_type*>(c_storage.data()), &offset);

        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        assert(err == cudaSuccess && "mul_cuda: CUDA error after synchronize.");

        return TensorImpl(c_storage, meta.aligned_shape_, out_dtype);
      });
    });
  });
}

TensorImpl div_cpu(const TensorImpl& a, const TensorImpl& b) {
  assert(false && "Not Implemented");
}

TensorImpl div_cuda(const TensorImpl& a, const TensorImpl& b) {
  detail::AlignUtil meta(a.shape(), b.shape());
  detail::cuda::OffsetUtil offset(a.shape(), b.shape(), a.strides(),
                                  b.strides(), meta.aligned_stride_,
                                  meta.aligned_shape_.size());
  DataType out_dtype = type_upcast(a.type(), b.type());
  return LMP_DISPATCH_ALL_TYPES(a.type(), [&] {
    using a_type_t = scalar_t;
    return LMP_DISPATCH_ALL_TYPES(b.type(), [&] {
      using b_type_t = scalar_t;
      return LMP_DISPATCH_ALL_TYPES(out_dtype, [&] {
        using out_type = scalar_t;
        Storage c_storage(meta.aligned_size_ * sizeof(out_type),
                          DeviceType::CUDA);

        ::lmp::tensor::detail::cuda::vecDiv<a_type_t, b_type_t, out_type>(
            meta.aligned_size_, static_cast<const a_type_t*>(a.data()),
            static_cast<const b_type_t*>(b.data()),
            static_cast<out_type*>(c_storage.data()), &offset);

        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        assert(err == cudaSuccess && "div_cuda: CUDA error after synchronize.");

        return TensorImpl(c_storage, meta.aligned_shape_, out_dtype);
      });
    });
  });
}

Tensor add(const Tensor& a, const Tensor& b) {
  assert(a.device() == b.device() && "Tensors are on different devices");
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      add_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a),
               *detail::UnsafeTensorAccessor::getImpl(b))));
}

Tensor sub(const Tensor& a, const Tensor& b) {
  assert(a.device() == b.device() && "Tensors are on different devices");
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      sub_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a),
               *detail::UnsafeTensorAccessor::getImpl(b))));
}

Tensor mul(const Tensor& a, const Tensor& b) {
  assert(a.device() == b.device() && "Tensors are on different devices");
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      mul_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a),
               *detail::UnsafeTensorAccessor::getImpl(b))));
}

Tensor div(const Tensor& a, const Tensor& b) {
  assert(a.device() == b.device() && "Tensors are on different devices");
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      div_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a),
               *detail::UnsafeTensorAccessor::getImpl(b))));
}

}  // namespace lmp::tensor::ops