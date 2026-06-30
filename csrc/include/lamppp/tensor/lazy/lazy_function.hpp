#pragma once
#include <memory>
#include <string>
#include <vector>

namespace lmp::tensor {
class TensorImpl;

struct LazyFunction {
  std::vector<std::shared_ptr<TensorImpl>> inputs;

  explicit LazyFunction(std::vector<std::shared_ptr<TensorImpl>> ins)
      : inputs(std::move(ins)) {}
  virtual ~LazyFunction() = default;

  virtual std::shared_ptr<TensorImpl> infer_output() const = 0;
  virtual void run_eager(TensorImpl& out) = 0;
  virtual std::string codegen_expr() const = 0;
  virtual bool is_fusible() const = 0;
};
}
