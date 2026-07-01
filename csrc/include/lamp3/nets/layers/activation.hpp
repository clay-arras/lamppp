#pragma once

#include "lamp3/autograd/variable.hpp"
#include "lamp3/nets/module.hpp"

namespace lmp::nets {

class ReLUImpl : public ModuleImpl {
 public:
  ReLUImpl() = default;
  autograd::Variable forward(const autograd::Variable& x) const;
};
LMP_DEFINE_MODULE(ReLU);

class SigmoidImpl : public ModuleImpl {
 public:
  SigmoidImpl() = default;
  autograd::Variable forward(const autograd::Variable& x) const;
};
LMP_DEFINE_MODULE(Sigmoid);

class TanhImpl : public ModuleImpl {
 public:
  TanhImpl() = default;
  autograd::Variable forward(const autograd::Variable& x) const;
};
LMP_DEFINE_MODULE(Tanh);

class SoftmaxImpl : public ModuleImpl {
 public:
  explicit SoftmaxImpl(ssize_t dim);
  autograd::Variable forward(const autograd::Variable& x) const;

 private:
  ssize_t dim_;
};
LMP_DEFINE_MODULE(Softmax);

}  // namespace lmp::nets