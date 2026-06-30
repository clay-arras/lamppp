#include "lamppp/tensor/lazy/record.hpp"

namespace lmp::tensor {

std::shared_ptr<TensorImpl> record(std::shared_ptr<LazyFunction> fn) {
  std::shared_ptr<TensorImpl> out = fn->infer_output();
  out->set_deferred(std::move(fn));
  return out;
}

}
