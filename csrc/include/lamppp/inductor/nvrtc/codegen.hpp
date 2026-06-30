#pragma once

#include <string>

namespace lmp::inductor {

struct FusedGraph;

/**
 * @brief Entry-point symbol of the generated kernel.
 *
 * @details Kernel ABI is (out, in0 .. in{K-1}, n): the output pointer first,
 * then boundary inputs in `FusedGraph::slot` order, then the element count.
 * The future launch piece loads this symbol and binds arguments in this order.
 */
inline constexpr const char* kFusedKernelName = "fused_kernel";

/**
 * @brief Generate CUDA source for the fused elementwise kernel of `g`.
 *
 * @details Emits one `__global__` function (`kFusedKernelName`) evaluating the
 * whole group element-wise over a flat 1-D index. Every value is declared with
 * its own real dtype; each op promotes its operands to the node's dtype.
 *
 * @pre Every node in `g.order` is a fusible elementwise op.
 */
std::string codegen_kernel(const FusedGraph& g);

}  // namespace lmp::inductor
