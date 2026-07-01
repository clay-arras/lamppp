#pragma once
#include <cstddef>
#include <vector>
#include "lamp3/tensor/data_type.hpp"

namespace lmp::tensor {
class TensorImpl;
namespace detail {

struct OpMeta {
  DataType dtype;
  size_t size;
  std::vector<size_t> shape;
  bool expand;
};

OpMeta infer_unary (const TensorImpl* a);
OpMeta infer_binary(const TensorImpl* a, const TensorImpl* b);
OpMeta infer_reduct(const TensorImpl* a, size_t axis);

}  // namespace detail
}  // namespace lmp::tensor
