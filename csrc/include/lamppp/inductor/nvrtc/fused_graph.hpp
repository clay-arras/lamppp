#pragma once

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "lamppp/tensor/lazy/lazy_function.hpp"
#include "lamppp/tensor/lazy/realize.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::inductor {

struct FusedGraph {
  std::vector<tensor::TensorImpl*> inputs;
  std::unordered_map<tensor::TensorImpl*, size_t> slot;

  std::vector<tensor::TensorImpl*> order;

  std::unordered_set<tensor::TensorImpl*> seen;

  tensor::TensorImpl* output = nullptr;
};

FusedGraph build_fused_graph(tensor::TensorImpl* root);

}
