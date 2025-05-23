#include "lamppp/tensor/cpu/unary.hpp"
#include "lamppp/tensor/cpu/kernels.hpp"

namespace lmp::tensor::detail::cpu {

template <typename PtrList, typename OpFn>
void vectorized_unary_kernel(PtrList ptr_, OpFn fn_, size_t i) {
  ptr_.set_Out(i, fn_(::std::get<1>(ptr_.fns)(ptr_.data[1], i)));
}

template <typename PtrList, typename OpFn>
void unary_kernel_launcher(PtrList ptr_, OpFn fn_, size_t size) {
#pragma omp \
    parallel for simd  // TODO: should guarantee alignment?? with aligned(a,out:64)?
  for (size_t i = 0; i < size; i++) {
    vectorized_unary_kernel(ptr_, fn_, i);
  }
}

template void unary_dispatch_handler<ExpFunctor>(UnaryMetaHandler&);
template void unary_dispatch_handler<LogFunctor>(UnaryMetaHandler&);
template void unary_dispatch_handler<SqrtFunctor>(UnaryMetaHandler&);
template void unary_dispatch_handler<AbsFunctor>(UnaryMetaHandler&);
template void unary_dispatch_handler<SinFunctor>(UnaryMetaHandler&);
template void unary_dispatch_handler<CosFunctor>(UnaryMetaHandler&);
template void unary_dispatch_handler<TanFunctor>(UnaryMetaHandler&);
template void unary_dispatch_handler<ClampFunctor>(UnaryMetaHandler&, Scalar&&,
                                                   Scalar&&);

}  // namespace lmp::tensor::detail::cpu