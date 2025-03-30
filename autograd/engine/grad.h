#ifndef _GRAD_H_
#define _GRAD_H_

#include <cmath>
#include <memory>
#include "engine.h"

class Value;

struct AddBackwardContext {
  std::shared_ptr<Value> self;
  std::shared_ptr<Value> other;
  std::shared_ptr<Value> out;

  AddBackwardContext(std::shared_ptr<Value> s, std::shared_ptr<Value> o,
                     std::shared_ptr<Value> out)
      : self(s), other(o), out(out) {}
};

struct MulBackwardContext {
  std::shared_ptr<Value> self;
  std::shared_ptr<Value> other;
  std::shared_ptr<Value> out;

  MulBackwardContext(std::shared_ptr<Value> s, std::shared_ptr<Value> o,
                     std::shared_ptr<Value> out)
      : self(s), other(o), out(out) {}
};

struct PowBackwardContext {
  std::shared_ptr<Value> self;
  std::shared_ptr<Value> other;
  std::shared_ptr<Value> out;

  PowBackwardContext(std::shared_ptr<Value> s, std::shared_ptr<Value> o,
                     std::shared_ptr<Value> out)
      : self(s), other(o), out(out) {}
};

struct ExpBackwardContext {
  std::shared_ptr<Value> self;
  std::shared_ptr<Value> out;

  ExpBackwardContext(std::shared_ptr<Value> s, std::shared_ptr<Value> out)
      : self(s), out(out) {}
};

struct LogBackwardContext {
  std::shared_ptr<Value> self;
  std::shared_ptr<Value> out;

  LogBackwardContext(std::shared_ptr<Value> s, std::shared_ptr<Value> out)
      : self(s), out(out) {}
};

struct ReluBackwardContext {
  std::shared_ptr<Value> self;
  std::shared_ptr<Value> out;

  ReluBackwardContext(std::shared_ptr<Value> s, std::shared_ptr<Value> out)
      : self(s), out(out) {}
};

void add_backward(void* ctx);
void mul_backward(void* ctx);
void pow_backward(void* ctx);
void exp_backward(void* ctx);
void log_backward(void* ctx);
void relu_backward(void* ctx);

#endif  // _GRAD_H_
