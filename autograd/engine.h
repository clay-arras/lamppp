#ifndef _ENGINE_H_
#define _ENGINE_H_

#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <unordered_set>
#include <algorithm>
#include <memory>
#include <cassert>
#include <vector>
#include "grad.h"
#include <cmath>
#include <initializer_list>

typedef void (*BackwardFn)(void*);

class Value : public std::enable_shared_from_this<Value> {
private:
  std::vector<std::shared_ptr<Value>> internalTopoSort();

public:
  double data;
  double grad;
  BackwardFn backward_fn = nullptr;
  void* backward_ctx = nullptr;
  std::unordered_set<std::shared_ptr<Value>> prev;

  Value(double data, std::unordered_set<std::shared_ptr<Value>> children = {},
        double grad = 0.0);
        
  void backward() {
    assert(backward_fn != nullptr);
    backward_fn(backward_ctx);
  }

  std::shared_ptr<Value> operator+(const std::shared_ptr<Value> &other);
  std::shared_ptr<Value> operator-(const std::shared_ptr<Value> &other);
  std::shared_ptr<Value> operator*(const std::shared_ptr<Value> &other);
  std::shared_ptr<Value> operator/(const std::shared_ptr<Value> &other);

  std::shared_ptr<Value> exp();
  std::shared_ptr<Value> log();
  std::shared_ptr<Value> tanh();
  std::shared_ptr<Value> relu();
  std::shared_ptr<Value> pow(std::shared_ptr<Value> other);

  void backprop();
  friend std::ostream &operator<<(std::ostream &os, const Value &obj);
};

std::shared_ptr<Value> operator+(const std::shared_ptr<Value> &a,
                                 const std::shared_ptr<Value> &b);
std::shared_ptr<Value> operator-(const std::shared_ptr<Value> &a,
                                 const std::shared_ptr<Value> &b);
std::shared_ptr<Value> operator*(const std::shared_ptr<Value> &a,
                                 const std::shared_ptr<Value> &b);
std::shared_ptr<Value> operator/(const std::shared_ptr<Value> &a,
                                 const std::shared_ptr<Value> &b);

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& a, const float b);
std::shared_ptr<Value> operator+(const float b, const std::shared_ptr<Value>& a);

std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a, const float b);
std::shared_ptr<Value> operator-(const float b, const std::shared_ptr<Value>& a);

std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& a, const float b);
std::shared_ptr<Value> operator*(const float b, const std::shared_ptr<Value>& a);

std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& a, const float b);
std::shared_ptr<Value> operator/(const float b, const std::shared_ptr<Value>& a);

std::shared_ptr<Value> exp(const std::shared_ptr<Value> &value);
std::shared_ptr<Value> log(const std::shared_ptr<Value> &value);
std::shared_ptr<Value> tanh(const std::shared_ptr<Value> &value);
std::shared_ptr<Value> relu(const std::shared_ptr<Value> &value);

#endif
