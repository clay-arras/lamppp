#pragma once
#include <memory>
#include <string>
#include <vector>

namespace lmp::tensor {
class TensorImpl;

// Fusion analogue of autograd::Function: describes a pending op.
struct LazyFunction {
  std::vector<std::shared_ptr<TensorImpl>> inputs;   // producers, kept alive

  explicit LazyFunction(std::vector<std::shared_ptr<TensorImpl>> ins)
      : inputs(std::move(ins)) {}
  virtual ~LazyFunction() = default;

  virtual std::shared_ptr<TensorImpl> infer_output() const = 0; // 0-byte impl + real meta
  virtual void run_eager(TensorImpl& out) = 0;                  // bridge to existing *_stub()
  virtual std::string codegen_expr() const = 0;                 // Part 2 seam (per-op RHS template)
  virtual bool is_fusible() const = 0;                          // Part 2 partition label
};
}  // namespace lmp::tensor
