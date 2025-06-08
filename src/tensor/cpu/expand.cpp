#include "lamppp/tensor/cpu/expand.hpp"
#include "lamppp/tensor/cpu/kernels.hpp"

namespace lmp::tensor::detail::cpu {

template <typename PtrList, typename OpFn>
void vectorized_expand_kernel(PtrList ptr_, OpFn fn_, size_t i,
                              const CPUOffsetUtil<kNArgs>* align) {
  ::std::array offsets = align->get(i);
  ptr_.set_Out(i, fn_(::std::get<1>(ptr_.fns)(ptr_.data[1], offsets[1]),
                      ::std::get<2>(ptr_.fns)(ptr_.data[2], offsets[2])));
}

template <typename PtrList, typename OpFn>
void expand_kernel_launcher(PtrList ptr_, OpFn fn_, size_t size,
                            const CPUOffsetUtil<kNArgs>* align) {
#pragma omp parallel for simd
  for (size_t i = 0; i < size; i++) {
    vectorized_expand_kernel(ptr_, fn_, i, align);
  }
}

template <template <typename> class OpFunctor, typename... Args>
void expand_dispatch_handler(ExpandMetaHandler& meta, Args&&... args) {
  LMP_DISPATCH_ALL_TYPES(meta.out().type(), [&] {
    using out_dtype_t = scalar_t;
    LMP_DISPATCH_ALL_TYPES(meta.in()[0]->type(), [&] {
      using arg1_dtype_t = scalar_t;
      LMP_DISPATCH_ALL_TYPES(meta.in()[1]->type(), [&] {
        using arg2_dtype_t = scalar_t;
        expand_kernel_launcher(
            internal::PtrPack<out_dtype_t, arg1_dtype_t, arg2_dtype_t>(
                static_cast<out_dtype_t*>(meta.out().data()),
                static_cast<arg1_dtype_t*>(meta.in()[0]->data()),
                static_cast<arg2_dtype_t*>(meta.in()[1]->data())),
            OpFunctor<out_dtype_t>(std::forward<Args>(args)...),
            meta.out().numel(),
            static_cast<const CPUOffsetUtil<kNArgs>*>(meta.offset()));
      });
    });
  });
}

template void expand_dispatch_handler<AddFunctor>(ExpandMetaHandler&);
template void expand_dispatch_handler<SubFunctor>(ExpandMetaHandler&);
template void expand_dispatch_handler<MulFunctor>(ExpandMetaHandler&);
template void expand_dispatch_handler<DivFunctor>(ExpandMetaHandler&);
template void expand_dispatch_handler<PowFunctor>(ExpandMetaHandler&);
template void expand_dispatch_handler<EqFunctor>(ExpandMetaHandler&);
template void expand_dispatch_handler<NeFunctor>(ExpandMetaHandler&);
template void expand_dispatch_handler<GeFunctor>(ExpandMetaHandler&);
template void expand_dispatch_handler<GtFunctor>(ExpandMetaHandler&);
template void expand_dispatch_handler<LeFunctor>(ExpandMetaHandler&);
template void expand_dispatch_handler<LtFunctor>(ExpandMetaHandler&);

}  // namespace lmp::tensor::detail::cpu