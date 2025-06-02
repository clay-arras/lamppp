#include "lamppp/tensor/native/memory_ops.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor::ops {

LMP_DEFINE_DISPATCH(copy_fn, copy_stub);
LMP_DEFINE_DISPATCH(empty_fn, empty_stub);
LMP_DEFINE_DISPATCH(fill_fn, fill_stub);
LMP_DEFINE_DISPATCH(resize_fn, resize_stub);

}  // namespace lmp::tensor::ops
