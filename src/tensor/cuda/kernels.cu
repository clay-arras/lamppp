#include "lamppp/common/macros.hpp"
#include "lamppp/tensor/cpu/meta_handler.hpp"
#include "lamppp/tensor/cuda/binary.cuh"
#include "lamppp/tensor/cuda/expand.cuh"
#include "lamppp/tensor/cuda/kernels.cuh"
#include "lamppp/tensor/cuda/matrix.cuh"
#include "lamppp/tensor/cuda/reduct.cuh"
#include "lamppp/tensor/cuda/unary.cuh"
#include "lamppp/tensor/cuda/conv.cuh"
#include "lamppp/tensor/native/conv_ops.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::detail::cuda {

#define DECLARE_EXPAND_OPS_CUDA(args) DECLARE_EXPAND_OPS_CUDA_HELPER args
#define DECLARE_EXPAND_OPS_CUDA_HELPER(op, functor)                \
  TensorImpl op##_cuda(const TensorImpl& a, const TensorImpl& b) { \
    TensorMetaHandler meta(&a, &b);                                \
    if (meta.expand()) {                                           \
      expand_dispatch_handler<functor>(meta);                      \
    } else {                                                       \
      binary_dispatch_handler<functor>(meta);                      \
    }                                                              \
    return meta.out();                                             \
  }

LMP_FOR_EACH_CARTESIAN_PRODUCT(
    DECLARE_EXPAND_OPS_CUDA,
    ((add, AddFunctor), (sub, SubFunctor), (mul, MulFunctor), (div, DivFunctor),
     (pow, PowFunctor), (eq, EqFunctor), (ne, NeFunctor), (le, LeFunctor),
     (lt, LtFunctor), (ge, GeFunctor), (gt, GtFunctor), ));

#define DECLARE_UNARY_OPS_CUDA(args) DECLARE_UNARY_OPS_CUDA_HELPER args
#define DECLARE_UNARY_OPS_CUDA_HELPER(op, functor) \
  TensorImpl op##_cuda(const TensorImpl& a) {      \
    TensorMetaHandler meta(&a);                    \
    unary_dispatch_handler<functor>(meta);         \
    return meta.out();                             \
  }

LMP_FOR_EACH_CARTESIAN_PRODUCT(DECLARE_UNARY_OPS_CUDA,
                               ((neg, NegFunctor), (log, LogFunctor),
                                (exp, ExpFunctor), (sqrt, SqrtFunctor),
                                (abs, AbsFunctor), (sin, SinFunctor),
                                (cos, CosFunctor), (tan, TanFunctor), ));

TensorImpl clamp_cuda(const TensorImpl& a, Scalar min_val, Scalar max_val) {
  TensorMetaHandler meta(&a);
  unary_dispatch_handler<ClampFunctor>(meta, min_val, max_val);
  return meta.out();
}

TensorImpl transpose_cuda(const TensorImpl& a) {
  LMP_CHECK(a.shape().size() == 2) << "Invalid argument, transpose can only be "
                                      "performed on matrices of dim 2";
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

TensorImpl matmul_cuda(const TensorImpl& a, const TensorImpl& b) {
  LMP_CHECK(a.shape().size() == 2 && b.shape().size() == 2)
      << "Both matrices must be 2D.";
  LMP_CHECK(a.shape()[1] == b.shape()[0])
      << "Incompatible matrix dimensions for multiplication.";

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

TensorImpl conv1d_cuda(const TensorImpl& input, const TensorImpl& kernel,
                     size_t stride, size_t padding, size_t dilation) {
  LMP_CHECK(input.shape().size() == 1 && kernel.shape().size() == 1)
      << "Both input and kernel must be 1D for conv1d.";
  LMP_CHECK(stride > 0) << "Stride must be positive.";
  LMP_CHECK(padding >= 0) << "Padding must be non-negative.";
  LMP_CHECK(dilation > 0) << "Dilation must be positive.";

  DataType out_dtype = type_upcast(input.type(), kernel.type());
  
  ssize_t out_length = ((input.shape()[0] + 2 * padding -
                      dilation * (kernel.shape()[0] - 1) - 1) /
                         stride) + 1;
  
  LMP_CHECK(out_length > 0) << "Invalid convolution parameters: output dimension is non-positive.";
  
  std::vector<size_t> out_shape{static_cast<size_t>(out_length)};

  return LMP_DISPATCH_ALL_TYPES(input.type(), [&] {
    using in_type_t = scalar_t;
    return LMP_DISPATCH_ALL_TYPES(kernel.type(), [&] {
      using kern_type_t = scalar_t;
      return LMP_DISPATCH_ALL_TYPES(out_dtype, [&] {
        using out_type_t = scalar_t;
        Storage c_storage(out_length * sizeof(out_type_t), DeviceType::CUDA);
        
        ::lmp::tensor::detail::cuda::cudaConv1d<in_type_t, kern_type_t, out_type_t>(
            static_cast<const in_type_t*>(input.data()),
            static_cast<const kern_type_t*>(kernel.data()),
            static_cast<out_type_t*>(c_storage.data()), 
            stride, padding, dilation, 
            input.shape().data(), kernel.shape().data(), out_shape.data());
        
        return TensorImpl(c_storage, out_shape, out_dtype);
      });
    });
  });
}

TensorImpl conv2d_cuda(const TensorImpl& input, const TensorImpl& kernel,
                     size_t stride, size_t padding, size_t dilation) {
  LMP_CHECK(input.shape().size() == 2 && kernel.shape().size() == 2)
      << "Both input and kernel must be 2D for conv2d.";
  LMP_CHECK(stride > 0) << "Stride must be positive.";
  LMP_CHECK(padding >= 0) << "Padding must be non-negative.";
  LMP_CHECK(dilation > 0) << "Dilation must be positive.";

  DataType out_dtype = type_upcast(input.type(), kernel.type());
  
  ssize_t out_h = ((input.shape()[0] + 2 * padding -
                  dilation * (kernel.shape()[0] - 1) - 1) /
                     stride) + 1;
  ssize_t out_w = ((input.shape()[1] + 2 * padding -
                  dilation * (kernel.shape()[1] - 1) - 1) /
                     stride) + 1;
  
  LMP_CHECK(out_h > 0 && out_w > 0) 
      << "Invalid convolution parameters: output dimensions are non-positive.";
  
  std::vector<size_t> out_shape{static_cast<size_t>(out_h), static_cast<size_t>(out_w)};

  return LMP_DISPATCH_ALL_TYPES(input.type(), [&] {
    using in_type_t = scalar_t;
    return LMP_DISPATCH_ALL_TYPES(kernel.type(), [&] {
      using kern_type_t = scalar_t;
      return LMP_DISPATCH_ALL_TYPES(out_dtype, [&] {
        using out_type_t = scalar_t;
        Storage c_storage(out_h * out_w * sizeof(out_type_t), DeviceType::CUDA);
        
        ::lmp::tensor::detail::cuda::cudaConv2d<in_type_t, kern_type_t, out_type_t>(
            static_cast<const in_type_t*>(input.data()),
            static_cast<const kern_type_t*>(kernel.data()),
            static_cast<out_type_t*>(c_storage.data()), 
            stride, padding, dilation, 
            input.shape().data(), kernel.shape().data(), out_shape.data());
        
        return TensorImpl(c_storage, out_shape, out_dtype);
      });
    });
  });
}

TensorImpl conv3d_cuda(const TensorImpl& input, const TensorImpl& kernel,
                     size_t stride, size_t padding, size_t dilation) {
  LMP_CHECK(input.shape().size() == 3 && kernel.shape().size() == 3)
      << "Both input and kernel must be 3D for conv3d.";
  LMP_CHECK(stride > 0) << "Stride must be positive.";
  LMP_CHECK(padding >= 0) << "Padding must be non-negative.";
  LMP_CHECK(dilation > 0) << "Dilation must be positive.";

  DataType out_dtype = type_upcast(input.type(), kernel.type());
  
  ssize_t out_d = ((input.shape()[0] + 2 * padding -
                  dilation * (kernel.shape()[0] - 1) - 1) /
                     stride) + 1;
  ssize_t out_h = ((input.shape()[1] + 2 * padding -
                  dilation * (kernel.shape()[1] - 1) - 1) /
                     stride) + 1;
  ssize_t out_w = ((input.shape()[2] + 2 * padding -
                  dilation * (kernel.shape()[2] - 1) - 1) /
                     stride) + 1;
  
  LMP_CHECK(out_d > 0 && out_h > 0 && out_w > 0) 
      << "Invalid convolution parameters: output dimensions are non-positive.";
  
  std::vector<size_t> out_shape{
    static_cast<size_t>(out_d), 
    static_cast<size_t>(out_h), 
    static_cast<size_t>(out_w)
  };

  return LMP_DISPATCH_ALL_TYPES(input.type(), [&] {
    using in_type_t = scalar_t;
    return LMP_DISPATCH_ALL_TYPES(kernel.type(), [&] {
      using kern_type_t = scalar_t;
      return LMP_DISPATCH_ALL_TYPES(out_dtype, [&] {
        using out_type_t = scalar_t;
        Storage c_storage(out_d * out_h * out_w * sizeof(out_type_t), DeviceType::CUDA);
        
        ::lmp::tensor::detail::cuda::cudaConv3d<in_type_t, kern_type_t, out_type_t>(
            static_cast<const in_type_t*>(input.data()),
            static_cast<const kern_type_t*>(kernel.data()),
            static_cast<out_type_t*>(c_storage.data()), 
            stride, padding, dilation, 
            input.shape().data(), kernel.shape().data(), out_shape.data());
        
        return TensorImpl(c_storage, out_shape, out_dtype);
      });
    });
  });
}

#define DECLARE_REDUCT_OPS_CUDA(args) DECLARE_REDUCT_OPS_CUDA_HELPER args
#define DECLARE_REDUCT_OPS_CUDA_HELPER(op, functor)        \
  TensorImpl op##_cuda(const TensorImpl& a, size_t axis) { \
    TensorMetaHandler meta(&a, axis);                      \
    reduct_dispatch_handler<functor>(meta, axis);          \
    return meta.out();                                     \
  }

LMP_FOR_EACH_CARTESIAN_PRODUCT(DECLARE_REDUCT_OPS_CUDA,
                               ((sum, SumFunctor), (max, MaxFunctor),
                                (min, MinFunctor), (prod, ProdFunctor), ));

LMP_REGISTER_DISPATCH(ops::add_stub, DeviceType::CUDA, add_cuda);
LMP_REGISTER_DISPATCH(ops::sub_stub, DeviceType::CUDA, sub_cuda);
LMP_REGISTER_DISPATCH(ops::mul_stub, DeviceType::CUDA, mul_cuda);
LMP_REGISTER_DISPATCH(ops::div_stub, DeviceType::CUDA, div_cuda);
LMP_REGISTER_DISPATCH(ops::pow_stub, DeviceType::CUDA, pow_cuda);
LMP_REGISTER_DISPATCH(ops::eq_stub, DeviceType::CUDA, eq_cuda);
LMP_REGISTER_DISPATCH(ops::ne_stub, DeviceType::CUDA, ne_cuda);
LMP_REGISTER_DISPATCH(ops::le_stub, DeviceType::CUDA, le_cuda);
LMP_REGISTER_DISPATCH(ops::lt_stub, DeviceType::CUDA, lt_cuda);
LMP_REGISTER_DISPATCH(ops::ge_stub, DeviceType::CUDA, ge_cuda);
LMP_REGISTER_DISPATCH(ops::gt_stub, DeviceType::CUDA, gt_cuda);

LMP_REGISTER_DISPATCH(ops::neg_stub, DeviceType::CUDA, neg_cuda);
LMP_REGISTER_DISPATCH(ops::log_stub, DeviceType::CUDA, log_cuda);
LMP_REGISTER_DISPATCH(ops::exp_stub, DeviceType::CUDA, exp_cuda);
LMP_REGISTER_DISPATCH(ops::sqrt_stub, DeviceType::CUDA, sqrt_cuda);
LMP_REGISTER_DISPATCH(ops::abs_stub, DeviceType::CUDA, abs_cuda);
LMP_REGISTER_DISPATCH(ops::sin_stub, DeviceType::CUDA, sin_cuda);
LMP_REGISTER_DISPATCH(ops::cos_stub, DeviceType::CUDA, cos_cuda);
LMP_REGISTER_DISPATCH(ops::tan_stub, DeviceType::CUDA, tan_cuda);
LMP_REGISTER_DISPATCH(ops::clamp_stub, DeviceType::CUDA, clamp_cuda);

LMP_REGISTER_DISPATCH(ops::matmul_stub, DeviceType::CUDA, matmul_cuda);
LMP_REGISTER_DISPATCH(ops::transpose_stub, DeviceType::CUDA, transpose_cuda);
LMP_REGISTER_DISPATCH(ops::conv1d_stub, DeviceType::CUDA, conv1d_cuda);
LMP_REGISTER_DISPATCH(ops::conv2d_stub, DeviceType::CUDA, conv2d_cuda);
LMP_REGISTER_DISPATCH(ops::conv3d_stub, DeviceType::CUDA, conv3d_cuda);

LMP_REGISTER_DISPATCH(ops::sum_stub, DeviceType::CUDA, sum_cuda);
LMP_REGISTER_DISPATCH(ops::max_stub, DeviceType::CUDA, max_cuda);
LMP_REGISTER_DISPATCH(ops::min_stub, DeviceType::CUDA, min_cuda);
LMP_REGISTER_DISPATCH(ops::prod_stub, DeviceType::CUDA, prod_cuda);

}  // namespace lmp::tensor::detail::cuda