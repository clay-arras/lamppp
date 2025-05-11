#include "cuda_backend.hpp"
#include "autograd/engine/cuda/basic_kern.cuh"
#include "autograd/engine/cuda/binary_kern.cuh"
#include "autograd/engine/cuda/matrix_kern.cuh"
#include "autograd/engine/cuda/reduct_kern.cuh"
#include "autograd/engine/cuda/unary_kern.cuh"
#include "autograd/engine/data_type.hpp"
#include "autograd/engine/device_type.hpp"
#include "autograd/engine/dispatch_type.hpp"
#include "autograd/engine/tensor_impl.hpp"

namespace autograd {

DataType CudaBackend::dtype_promotion_(DataType a_type, DataType b_type) {
  return static_cast<DataType>(
      std::max(static_cast<uint8_t>(a_type), static_cast<uint8_t>(b_type)));
}

TensorImpl CudaBackend::add(const TensorImpl& a, const TensorImpl& b) {
  assert(a.size() == b.size() && "Size mismatch");
  assert(a.shape() == b.shape() && "Shape mismatch");

  // NOTE: this is absolutely horrible
  DataType out_dtype = dtype_promotion_(a.type(), b.type());
  return DISPATCH_ALL_TYPES(a.type(), [&] {
    using a_type_t = scalar_t;
    return DISPATCH_ALL_TYPES(b.type(), [&] {
      using b_type_t = scalar_t;
      return DISPATCH_ALL_TYPES(out_dtype, [&] {
        using out_type = scalar_t;
        Storage c(a.size() * sizeof(out_type), DeviceType::CUDA);
        vecAdd<a_type_t, b_type_t, out_type>(
            a.size(), static_cast<const a_type_t*>(a.data()),
            static_cast<const b_type_t*>(b.data()),
            static_cast<out_type*>(c.data()));
        return TensorImpl(c, a.shape(), out_dtype);
      });
    });
  });
}

TensorImpl CudaBackend::sub(const TensorImpl& a, const TensorImpl& b) {
  assert(a.size() == b.size() && "Size mismatch");
  assert(a.shape() == b.shape() && "Shape mismatch");

  DataType out_dtype = dtype_promotion_(a.type(), b.type());
  return DISPATCH_ALL_TYPES(a.type(), [&] {
    using a_type_t = scalar_t;
    return DISPATCH_ALL_TYPES(b.type(), [&] {
      using b_type_t = scalar_t;
      return DISPATCH_ALL_TYPES(out_dtype, [&] {
        using out_type = scalar_t;
        Storage c(a.size() * sizeof(out_type), DeviceType::CUDA);
        vecSub<a_type_t, b_type_t, out_type>(
            a.size(), static_cast<const a_type_t*>(a.data()),
            static_cast<const b_type_t*>(b.data()),
            static_cast<out_type*>(c.data()));
        return TensorImpl(c, a.shape(), out_dtype);
      });
    });
  });
}

TensorImpl CudaBackend::mul(const TensorImpl& a, const TensorImpl& b) {
  assert(a.size() == b.size() && "Size mismatch");
  assert(a.shape() == b.shape() && "Shape mismatch");

  DataType out_dtype = dtype_promotion_(a.type(), b.type());
  return DISPATCH_ALL_TYPES(a.type(), [&] {
    using a_type_t = scalar_t;
    return DISPATCH_ALL_TYPES(b.type(), [&] {
      using b_type_t = scalar_t;
      return DISPATCH_ALL_TYPES(out_dtype, [&] {
        using out_type = scalar_t;
        Storage c(a.size() * sizeof(out_type), DeviceType::CUDA);
        vecMul<a_type_t, b_type_t, out_type>(
            a.size(), static_cast<const a_type_t*>(a.data()),
            static_cast<const b_type_t*>(b.data()),
            static_cast<out_type*>(c.data()));
        return TensorImpl(c, a.shape(), out_dtype);
      });
    });
  });
}

TensorImpl CudaBackend::div(const TensorImpl& a, const TensorImpl& b) {
  assert(a.size() == b.size() && "Size mismatch");
  assert(a.shape() == b.shape() && "Shape mismatch");

  DataType out_dtype = dtype_promotion_(a.type(), b.type());
  return DISPATCH_ALL_TYPES(a.type(), [&] {
    using a_type_t = scalar_t;
    return DISPATCH_ALL_TYPES(b.type(), [&] {
      using b_type_t = scalar_t;
      return DISPATCH_ALL_TYPES(out_dtype, [&] {
        using out_type = scalar_t;
        Storage c(a.size() * sizeof(out_type), DeviceType::CUDA);
        vecDiv<a_type_t, b_type_t, out_type>(
            a.size(), static_cast<const a_type_t*>(a.data()),
            static_cast<const b_type_t*>(b.data()),
            static_cast<out_type*>(c.data()));
        return TensorImpl(c, a.shape(), out_dtype);
      });
    });
  });
}

TensorImpl CudaBackend::log(const TensorImpl& a) {
  return DISPATCH_ALL_TYPES(a.type(), [&] {
    using scalar_t_ = scalar_t;
    Storage c(a.size() * sizeof(scalar_t_), DeviceType::CUDA);
    vecLog<scalar_t_>(a.size(), static_cast<const scalar_t_*>(a.data()),
                      static_cast<scalar_t_*>(c.data()));
    return TensorImpl(c, a.shape(), a.type());
  });
}

TensorImpl CudaBackend::exp(const TensorImpl& a) {
  return DISPATCH_ALL_TYPES(a.type(), [&] {
    using scalar_t_ = scalar_t;
    Storage c(a.size() * sizeof(scalar_t_), DeviceType::CUDA);
    vecExp<scalar_t_>(a.size(), static_cast<const scalar_t_*>(a.data()),
                      static_cast<scalar_t_*>(c.data()));
    return TensorImpl(c, a.shape(), a.type());
  });
}

TensorImpl CudaBackend::relu(const TensorImpl& a) {
  return DISPATCH_ALL_TYPES(a.type(), [&] {
    using scalar_t_ = scalar_t;
    Storage c(a.size() * sizeof(scalar_t_), DeviceType::CUDA);
    vecRelu<scalar_t_>(a.size(), static_cast<const scalar_t_*>(a.data()),
                       static_cast<scalar_t_*>(c.data()));
    return TensorImpl(c, a.shape(), a.type());
  });
}

TensorImpl CudaBackend::matmul(const TensorImpl& a, const TensorImpl& b) {
  assert(a.shape().size() == 2 && b.shape().size() == 2 &&
         "Invalid argument, matrix multiplication can only be performed on "
         "matrices of dim 2");
  assert(a.shape()[1] == b.shape()[0] &&
         "Invalid argument, the second dim of the first matrix must equal the "
         "first dim of the second matrix");

  size_t m = a.shape()[0];
  size_t n = b.shape()[1];
  size_t k = a.shape()[1];

  DataType out_dtype = dtype_promotion_(a.type(), b.type());

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

TensorImpl CudaBackend::transpose(const TensorImpl& a) {
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

TensorImpl CudaBackend::equal(const TensorImpl& a, const TensorImpl& b) {
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

TensorImpl CudaBackend::not_equal(const TensorImpl& a, const TensorImpl& b) {
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

TensorImpl CudaBackend::greater_equal(const TensorImpl& a,
                                      const TensorImpl& b) {
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

TensorImpl CudaBackend::less_equal(const TensorImpl& a, const TensorImpl& b) {
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

TensorImpl CudaBackend::greater(const TensorImpl& a, const TensorImpl& b) {
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

TensorImpl CudaBackend::less(const TensorImpl& a, const TensorImpl& b) {
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

TensorImpl CudaBackend::sum(const TensorImpl& a,
                            size_t axis) {  // NOTE: returning stub
  Storage c(a.size() * sizeof(bool), DeviceType::CUDA);
  return TensorImpl(c, a.shape(), a.type());
}

TensorImpl CudaBackend::max(const TensorImpl& a, size_t axis) {
  Storage c(a.size() * sizeof(bool), DeviceType::CUDA);
  return TensorImpl(c, a.shape(), a.type());
}

}  // namespace autograd