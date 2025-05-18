#include "lamppp/tensor/cuda/unary.cuh"
#include "lamppp/tensor/cuda/utils.cuh"
#include "lamppp/tensor/dispatch_stub.hpp"
#include "lamppp/tensor/functions/unary_ops.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::detail::cuda {

TensorImpl log_cuda(const TensorImpl& a) {
  internal::TensorMetaHandler meta({a});
  meta.handle_unary_op();
  unary_dispatch_handler<LogFunctor>(meta);
  return meta.out();
}

LMP_REGISTER_DISPATCH(ops::log_stub, DeviceType::CUDA, log_cuda);

}  // namespace lmp::tensor::detail::cuda