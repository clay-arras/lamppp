#include "lamppp/tensor/lazy/record.hpp"

#include "lamppp/tensor/lazy/lazy_function.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::tensor {

std::shared_ptr<TensorImpl> record(std::shared_ptr<LazyFunction> fn) {
  std::shared_ptr<TensorImpl> out = fn->infer_output();  // 0-byte impl + real meta (§3)
  out->set_deferred(std::move(fn));                      // attach pending op
  return out;
}

}  // namespace lmp::tensor
