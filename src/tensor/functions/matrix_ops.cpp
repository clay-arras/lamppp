#include "include/lamppp/tensor/functions/matrix_ops.hpp"
#include "include/lamppp/tensor/cuda/matrix_kern.cuh"
#include "include/lamppp/tensor/data_type.hpp"
#include "include/lamppp/tensor/tensor_impl.hpp"

namespace autograd {

DEFINE_DISPATCH(matmul_stub);
DEFINE_DISPATCH(transpose_stub);

TensorImpl matmul_cpu(const TensorImpl& a, const TensorImpl& b) {
  assert(false && "Not Implemented");
}

TensorImpl matmul_cuda(const TensorImpl& a, const TensorImpl& b) {
  assert(a.shape().size() == 2 && b.shape().size() == 2 &&
         "Invalid argument, matrix multiplication can only be performed on "
         "matrices of dim 2");
  assert(a.shape()[1] == b.shape()[0] &&
         "Invalid argument, the second dim of the first matrix must equal the "
         "first dim of the second matrix");

  size_t m = a.shape()[0];
  size_t n = b.shape()[1];
  size_t k = a.shape()[1];

  DataType out_dtype = type_upcast(a.type(), b.type());

  return DISPATCH_ALL_TYPES(a.type(), [&] {
    using a_type_t = scalar_t;
    return DISPATCH_ALL_TYPES(b.type(), [&] {
      using b_type_t = scalar_t;
      return DISPATCH_ALL_TYPES(out_dtype, [&] {
        using out_type_t = scalar_t;
        Storage c_storage(m * n * sizeof(out_type_t), DeviceType::CUDA);
        cudaMatMul<a_type_t, b_type_t, out_type_t>(
            static_cast<const a_type_t*>(a.data()),
            static_cast<const b_type_t*>(b.data()),
            static_cast<out_type_t*>(c_storage.data()), m, n, k);
        return TensorImpl(c_storage, {m, n}, out_dtype);
      });
    });
  });
}

TensorImpl transpose_cpu(const TensorImpl& a) {
  assert(false && "Not Implemented");
  return TensorImpl(Storage(0, DeviceType::CPU), {}, DataType::Float32);
}

TensorImpl transpose_cuda(const TensorImpl& a) {
  assert(
      a.shape().size() == 2 &&
      "Invalid argument, transpose can only be performed on matrices of dim 2");
  size_t m = a.shape()[0];
  size_t n = a.shape()[1];

  DataType out_dtype = a.type();

  return DISPATCH_ALL_TYPES(a.type(), [&] {
    using scalar_t_ = scalar_t;
    Storage c_storage(m * n * sizeof(scalar_t_), DeviceType::CUDA);
    cudaTranspose<scalar_t_>(static_cast<const scalar_t_*>(a.data()),
                             static_cast<scalar_t_*>(c_storage.data()), m, n);
    return TensorImpl(c_storage, {n, m}, out_dtype);
  });
}

Tensor matmul(const Tensor& a, const Tensor& b) {
  assert(a.device() == b.device() && "Tensors must be on the same device");
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      matmul_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a),
                  *detail::UnsafeTensorAccessor::getImpl(b))));
}

Tensor transpose(const Tensor& a) {
  return detail::UnsafeTensorAccessor::fromImpl(std::make_shared<TensorImpl>(
      transpose_stub(a.device(), *detail::UnsafeTensorAccessor::getImpl(a))));
}

}  // namespace autograd
