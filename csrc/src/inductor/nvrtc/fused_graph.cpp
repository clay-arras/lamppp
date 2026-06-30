#include "lamppp/inductor/nvrtc/fused_graph.hpp"

#include "lamppp/tensor/lazy/lazy_function.hpp"
#include "lamppp/tensor/lazy/realize.hpp"
#include "lamppp/tensor/tensor_impl.hpp"

namespace lmp::inductor {
namespace {

// Classify a NON-root node into g.
void visit(tensor::TensorImpl* n, FusedGraph& g) {
  if (g.seen.count(n))
    return;  // DAG-safe + diamond-safe
  g.seen.insert(n);

  // Interior iff still deferred AND fusible; otherwise it's a boundary input.
  if (!n->is_deferred() || !n->lazy_op()->is_fusible()) {
    tensor::realize(n);  // force boundary input to a realized leaf
    if (!g.slot.count(n)) {
      g.slot[n] = g.inputs.size();
      g.inputs.push_back(n);
    }
    return;
  }

  for (const std::shared_ptr<tensor::TensorImpl>& in : n->lazy_op()->inputs)
    visit(in.get(), g);
  g.order.push_back(n);
}

}  // namespace

FusedGraph build_fused_graph(tensor::TensorImpl* root) {
  FusedGraph g;
  g.output = root;
  g.seen.insert(root);  // root is the realization target -- always interior
  for (const std::shared_ptr<tensor::TensorImpl>& in : root->lazy_op()->inputs)
    visit(in.get(), g);
  g.order.push_back(root);  // root evaluated last
  return g;
}

}  // namespace lmp::inductor
