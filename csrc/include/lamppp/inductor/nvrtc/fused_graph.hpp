#pragma once

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace lmp::tensor {
class TensorImpl;
}

namespace lmp::inductor {

/**
 * @brief A maximal group of fusible deferred ops with its realized boundary.
 *
 * @details The partition of a lazily-captured graph that bottoms out at
 * realized leaves: interior nodes are still-deferred fusible ops in
 * leaves-first evaluation order, and the boundary `inputs` are the realized
 * tensors feeding the group. `slot` is the inverse of `inputs`, giving each
 * boundary input a stable index for kernel-argument binding.
 *
 * @see build_fused_graph
 */
struct FusedGraph {
  // Boundary inputs feeding the group: realized leaves. Two views of one bijection.
  std::vector<tensor::TensorImpl*> inputs;  // index -> impl (ordered)
  std::unordered_map<tensor::TensorImpl*, size_t> slot;  // impl -> index

  // Interior op-nodes in evaluation order (leaves-first; root last).
  std::vector<tensor::TensorImpl*> order;

  // Classification memo: every node already placed (interior or input).
  std::unordered_set<tensor::TensorImpl*> seen;

  // Result of the group. v0: exactly one (== root). Seam for future multi-output.
  tensor::TensorImpl* output = nullptr;
};

/**
 * @brief Partition the deferred graph rooted at `root` into a `FusedGraph`.
 *
 * @details Walks the operand DAG from `root`, classifying each non-root node as
 * interior (still deferred and fusible) or boundary input (realized as a side
 * effect). The root is always interior. DAG- and diamond-safe via the `seen`
 * memo.
 */
FusedGraph build_fused_graph(tensor::TensorImpl* root);

}  // namespace lmp::inductor
