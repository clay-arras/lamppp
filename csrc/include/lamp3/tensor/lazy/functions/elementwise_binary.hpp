#pragma once
#include <memory>
#include <string>
#include <vector>

#include "lamp3/tensor/infer_meta.hpp"
#include "lamp3/tensor/lazy/lazy_function.hpp"
#include "lamp3/tensor/native/expand_ops.hpp"
#include "lamp3/tensor/storage.hpp"
#include "lamp3/tensor/tensor_impl.hpp"

namespace lmp::tensor {

struct ElementwiseBinaryFn : LazyFunction {
  using LazyFunction::LazyFunction;
  std::shared_ptr<TensorImpl> infer_output() const override;
  bool is_fusible() const override;
};

struct AddFn : ElementwiseBinaryFn {
  using ElementwiseBinaryFn::ElementwiseBinaryFn;
  void run_eager(TensorImpl& out) override;
  std::string codegen_expr() const override { return "{0} + {1}"; }
};

struct SubFn : ElementwiseBinaryFn {
  using ElementwiseBinaryFn::ElementwiseBinaryFn;
  void run_eager(TensorImpl& out) override;
  std::string codegen_expr() const override { return "{0} - {1}"; }
};

struct MulFn : ElementwiseBinaryFn {
  using ElementwiseBinaryFn::ElementwiseBinaryFn;
  void run_eager(TensorImpl& out) override;
  std::string codegen_expr() const override { return "{0} * {1}"; }
};

struct DivFn : ElementwiseBinaryFn {
  using ElementwiseBinaryFn::ElementwiseBinaryFn;
  void run_eager(TensorImpl& out) override;
  std::string codegen_expr() const override { return "{0} / {1}"; }
};

struct PowFn : ElementwiseBinaryFn {
  using ElementwiseBinaryFn::ElementwiseBinaryFn;
  void run_eager(TensorImpl& out) override;
  std::string codegen_expr() const override { return "pow({0}, {1})"; }
};

struct EqFn : ElementwiseBinaryFn {
  using ElementwiseBinaryFn::ElementwiseBinaryFn;
  void run_eager(TensorImpl& out) override;
  std::string codegen_expr() const override { return "{0} == {1}"; }
};

struct NeFn : ElementwiseBinaryFn {
  using ElementwiseBinaryFn::ElementwiseBinaryFn;
  void run_eager(TensorImpl& out) override;
  std::string codegen_expr() const override { return "{0} != {1}"; }
};

struct GeFn : ElementwiseBinaryFn {
  using ElementwiseBinaryFn::ElementwiseBinaryFn;
  void run_eager(TensorImpl& out) override;
  std::string codegen_expr() const override { return "{0} >= {1}"; }
};

struct LeFn : ElementwiseBinaryFn {
  using ElementwiseBinaryFn::ElementwiseBinaryFn;
  void run_eager(TensorImpl& out) override;
  std::string codegen_expr() const override { return "{0} <= {1}"; }
};

struct GtFn : ElementwiseBinaryFn {
  using ElementwiseBinaryFn::ElementwiseBinaryFn;
  void run_eager(TensorImpl& out) override;
  std::string codegen_expr() const override { return "{0} > {1}"; }
};

struct LtFn : ElementwiseBinaryFn {
  using ElementwiseBinaryFn::ElementwiseBinaryFn;
  void run_eager(TensorImpl& out) override;
  std::string codegen_expr() const override { return "{0} < {1}"; }
};

}
