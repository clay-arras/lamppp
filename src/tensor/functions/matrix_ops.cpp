#include "lamppp/tensor/functions/matrix_ops.hpp"
#include "lamppp/tensor/cpu/matrix.hpp"
#include "lamppp/tensor/cuda/matrix.cuh"
#include "lamppp/tensor/data_type.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::ops {

LMP_DEFINE_DISPATCH(matmul_fn, matmul_stub);
LMP_DEFINE_DISPATCH(transpose_fn, transpose_stub);

TensorImpl matmul_cpu(const TensorImpl& a, const TensorImpl& b) {
  LMP_CHECK(a.shape().size() == 2 && b.shape().size() == 2,
            "Both matrices must be 2D.");
  LMP_CHECK(a.shape()[1] == b.shape()[0],
            "Incompatible matrix dimensions for multiplication.");

  size_t m = a.shape()[0];
  size_t n = b.shape()[1];
  size_t k = a.shape()[1];

  DataType out_dtype = type_upcast(a.type(), b.type());

  return LMP_DISPATCH_ALL_TYPES(a.type(), [&] {
    using a_type_t = scalar_t;
    return LMP_DISPATCH_ALL_TYPES(b.type(), [&] {
      using b_type_t = scalar_t;
      return LMP_DISPATCH_ALL_TYPES(out_dtype, [&] {
        using out_type_t = scalar_t;
        Storage c_storage(m * n * sizeof(out_type_t), DeviceType::CPU);
        ::lmp::tensor::detail::cpu::cpuMatMul<a_type_t, b_type_t, out_type_t>(
            static_cast<const a_type_t*>(a.data()),
            static_cast<const b_type_t*>(b.data()),
            static_cast<out_type_t*>(c_storage.data()), m, n, k);
        return TensorImpl(c_storage, {m, n}, out_dtype);
      });
    });
  });
}

TensorImpl matmul_cuda(const TensorImpl& a, const TensorImpl& b) {
  LMP_CHECK(a.shape().size() == 2 && b.shape().size() == 2,
            "Both matrices must be 2D.");
  LMP_CHECK(a.shape()[1] == b.shape()[0],
            "Incompatible matrix dimensions for multiplication.");

  size_t m = a.shape()[0];
  size_t n = b.shape()[1];
  size_t k = a.shape()[1];

  DataType out_dtype = type_upcast(a.type(), b.type());

  return LMP_DISPATCH_ALL_TYPES(a.type(), [&] {
    using a_type_t = scalar_t;
    return LMP_DISPATCH_ALL_TYPES(b.type(), [&] {
      using b_type_t = scalar_t;
      return LMP_DISPATCH_ALL_TYPES(out_dtype, [&] {
        using out_type_t = scalar_t;
        Storage c_storage(m * n * sizeof(out_type_t), DeviceType::CUDA);
        ::lmp::tensor::detail::cuda::cudaMatMul<a_type_t, b_type_t, out_type_t>(
            static_cast<const a_type_t*>(a.data()),
            static_cast<const b_type_t*>(b.data()),
            static_cast<out_type_t*>(c_storage.data()), m, n, k);
        return TensorImpl(c_storage, {m, n}, out_dtype);
      });
    });
  });
}

TensorImpl transpose_cpu(const TensorImpl& a) {
  LMP_CHECK(
      a.shape().size() == 2,
      "Invalid argument, transpose can only be performed on matrices of dim 2");
  size_t m = a.shape()[0];
  size_t n = a.shape()[1];

  DataType out_dtype = a.type();

  return LMP_DISPATCH_ALL_TYPES(a.type(), [&] {
    Storage c_storage(m * n * sizeof(scalar_t), DeviceType::CPU);
    ::lmp::tensor::detail::cpu::cpuTranspose<scalar_t>(
        static_cast<const scalar_t*>(a.data()),
        static_cast<scalar_t*>(c_storage.data()), m, n);
    return TensorImpl(c_storage, {n, m}, out_dtype);
  });
}

TensorImpl transpose_cuda(const TensorImpl& a) {
  LMP_CHECK(
      a.shape().size() == 2,
      "Invalid argument, transpose can only be performed on matrices of dim 2");
  size_t m = a.shape()[0];
  size_t n = a.shape()[1];

  DataType out_dtype = a.type();

  return LMP_DISPATCH_ALL_TYPES(a.type(), [&] {
    Storage c_storage(m * n * sizeof(scalar_t), DeviceType::CUDA);
    ::lmp::tensor::detail::cuda::cudaTranspose<scalar_t>(
        static_cast<const scalar_t*>(a.data()),
        static_cast<scalar_t*>(c_storage.data()), m, n);
    return TensorImpl(c_storage, {n, m}, out_dtype);
  });
}

Tensor matmul(const Tensor& a, const Tensor& b) {
  LMP_CHECK(a.device() == b.device(), "Tensors must be on the same device");
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      matmul_stub()(a.device(), *detail::UnsafeTensorAccessor::getImpl(a),
                    *detail::UnsafeTensorAccessor::getImpl(b))));
}

Tensor transpose(const Tensor& a) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      transpose_stub()(a.device(), *detail::UnsafeTensorAccessor::getImpl(a))));
}

}  // namespace lmp::tensor::ops
