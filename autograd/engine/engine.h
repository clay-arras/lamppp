#ifndef _ENGINE_H_
#define _ENGINE_H_

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

void add_backward(void* ctx);
void mul_backward(void* ctx);
void pow_backward(void* ctx);
void exp_backward(void* ctx);
void log_backward(void* ctx);
void relu_backward(void* ctx);
void tanh_backward(void* ctx);

using BackwardFn = void (*)(void*);

class Value : public std::enable_shared_from_this<Value> {
 private:
  std::vector<std::shared_ptr<Value>> internalTopoSort();

 public:
  double data;
  double grad;
  bool requires_grad;
  BackwardFn backward_fn = nullptr;
  void* backward_ctx = nullptr;
  std::vector<std::shared_ptr<Value>> prev;

  explicit Value(double data,
                bool requires_grad = false,
                std::vector<std::shared_ptr<Value>> children = {},
                double grad = 0.0);

  void backward() const {
    assert(backward_fn != nullptr);
    backward_fn(backward_ctx);
  }

  std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& other);
  std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& other);
  std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& other);
  std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& other);

  std::shared_ptr<Value> exp();
  std::shared_ptr<Value> log();
  std::shared_ptr<Value> tanh();
  std::shared_ptr<Value> relu();
  std::shared_ptr<Value> pow(const std::shared_ptr<Value>& other);

  void backprop();
  friend std::ostream& operator<<(std::ostream& os, const Value& obj);
};

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& a,
                                 const std::shared_ptr<Value>& b);
std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a,
                                 const std::shared_ptr<Value>& b);
std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& a,
                                 const std::shared_ptr<Value>& b);
std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& a,
                                 const std::shared_ptr<Value>& b);

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& a, float b);
std::shared_ptr<Value> operator+(float b, const std::shared_ptr<Value>& a);

std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a, float b);
std::shared_ptr<Value> operator-(float b, const std::shared_ptr<Value>& a);

std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& a, float b);
std::shared_ptr<Value> operator*(float b, const std::shared_ptr<Value>& a);

std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& a, float b);
std::shared_ptr<Value> operator/(float b, const std::shared_ptr<Value>& a);

std::shared_ptr<Value> exp(const std::shared_ptr<Value>& value);
std::shared_ptr<Value> log(const std::shared_ptr<Value>& value);
std::shared_ptr<Value> tanh(const std::shared_ptr<Value>& value);
std::shared_ptr<Value> relu(const std::shared_ptr<Value>& value);

#endif
