#include <algorithm>
#include <functional>
#include <memory>
#include <utility>
#include <unordered_set>

#include "autograd/engine/engine.h"
#include "autograd/engine/grad.h"
#include "autograd/engine/value_pool.h"

ValueMemoryPool Value::pool_(1000); 

Value::Value(double data, bool requires_grad, std::vector<std::shared_ptr<Value>> children,
             double grad)
    : data(data), grad(grad), prev(std::move(children)), requires_grad(requires_grad) {}
  
Value::Value(const Value& other) : Value(other.data, other.requires_grad, other.prev, other.grad) {}

void Value::destroy(Value* ptr) {
  ptr->~Value();
  pool_.deallocate(static_cast<void*>(ptr));
}

std::shared_ptr<Value> Value::create(const Value& value) {
  void* raw_memory = pool_.allocate();
  auto* block = new (raw_memory) Value(value);

  std::shared_ptr<Value> alloc(block, destroy);
  return alloc;
}

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& a,
                                 const std::shared_ptr<Value>& b) {
  return a->operator+(b);
}

std::shared_ptr<Value> Value::operator+(const std::shared_ptr<Value>& other) {
  if (!this->requires_grad && !other->requires_grad) {
    return Value::create(Value(this->data + other->data));
  }
  std::shared_ptr<Value> out = Value::create(Value(
      this->data + other->data, true,
      std::vector<std::shared_ptr<Value>>{shared_from_this(), other}));

  auto* ctx = new AddBackwardContext(shared_from_this(), other, out);
  out->backward_fn = &add_backward;
  out->backward_ctx = static_cast<void*>(ctx);

  return out;
}

std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a,
                                 const std::shared_ptr<Value>& b) {
  return a->operator-(b);
}

std::shared_ptr<Value> Value::operator-(const std::shared_ptr<Value>& other) {
  return shared_from_this() +
         (other * Value::create(Value(static_cast<double>(-1))));
}

std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& a,
                                 const std::shared_ptr<Value>& b) {
  return a->operator*(b);
}

std::shared_ptr<Value> Value::operator*(const std::shared_ptr<Value>& other) {
  if (!this->requires_grad && !other->requires_grad) {
    return Value::create(Value(this->data * other->data));
  }
  std::shared_ptr<Value> out = Value::create(Value(
      this->data * other->data, true,
      std::vector<std::shared_ptr<Value>>{shared_from_this(), other}));

  auto* ctx = new MulBackwardContext(shared_from_this(), other, out);
  out->backward_fn = &mul_backward;
  out->backward_ctx = static_cast<void*>(ctx);

  return out;
}

std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& a,
                                 const std::shared_ptr<Value>& b) {
  return a->operator/(b);
}

std::shared_ptr<Value> Value::operator/(const std::shared_ptr<Value>& other) {
  return shared_from_this() *
         other->pow(Value::create(Value(static_cast<double>(-1))));
}

std::shared_ptr<Value> Value::pow(const std::shared_ptr<Value>& other) {
  if (!this->requires_grad && !other->requires_grad) {
    return Value::create(Value(std::pow(this->data, other->data)));
  }
  std::shared_ptr<Value> out = Value::create(Value(
      std::pow(this->data, other->data), true,
      std::vector<std::shared_ptr<Value>>{shared_from_this(), other}));
  auto* ctx = new PowBackwardContext(shared_from_this(), other, out);
  out->backward_fn = &pow_backward;
  out->backward_ctx = static_cast<void*>(ctx);

  return out;
}

std::shared_ptr<Value> Value::exp() {
  if (!this->requires_grad) {
    return Value::create(Value(std::exp(this->data)));
  }
  std::shared_ptr<Value> out = Value::create(Value(
      std::exp(this->data), true,
      std::vector<std::shared_ptr<Value>>{shared_from_this()}));

  auto* ctx = new ExpBackwardContext(shared_from_this(), out);
  out->backward_fn = &exp_backward;
  out->backward_ctx = static_cast<void*>(ctx);

  return out;
}

std::shared_ptr<Value> exp(const std::shared_ptr<Value>& value) {
  return value->exp();
}

std::shared_ptr<Value> Value::log() {
  assert(this->data > 0);
  if (!this->requires_grad) {
    return Value::create(Value(std::log(this->data)));
  }
  std::shared_ptr<Value> out = Value::create(Value(
      std::log(this->data), true, 
      std::vector<std::shared_ptr<Value>>{shared_from_this()}));

  auto* ctx = new LogBackwardContext(shared_from_this(), out);
  out->backward_fn = &log_backward;
  out->backward_ctx = static_cast<void*>(ctx);

  return out;
}

std::shared_ptr<Value> log(const std::shared_ptr<Value>& value) {
  return value->log();
}

std::shared_ptr<Value> Value::tanh() {
  if (!this->requires_grad) {
    return Value::create(Value(std::tanh(this->data)));
  }
  std::shared_ptr<Value> out = Value::create(Value(
      std::tanh(this->data), true, std::vector<std::shared_ptr<Value>>{shared_from_this()}));

  auto* ctx = new TanhBackwardContext(shared_from_this(), out);
  out->backward_fn = &tanh_backward;
  out->backward_ctx = static_cast<void*>(ctx);

  return out;
}

std::shared_ptr<Value> tanh(const std::shared_ptr<Value>& value) {
  return value->tanh();
}

std::shared_ptr<Value> Value::relu() {
  if (!this->requires_grad) {
    return Value::create(Value(std::max(0.0, this->data)));
  }
  std::shared_ptr<Value> out = Value::create(Value(
      std::max(0.0, this->data), true, std::vector<std::shared_ptr<Value>>{shared_from_this()}));

  auto* ctx = new ReluBackwardContext(shared_from_this(), out);
  out->backward_fn = &relu_backward;
  out->backward_ctx = static_cast<void*>(ctx);

  return out;
}

std::shared_ptr<Value> relu(const std::shared_ptr<Value>& value) {
  return value->relu();
}

void Value::backprop() {
  std::vector<std::shared_ptr<Value>> topo = internalTopoSort();
  this->grad = 1.0;
  for (const std::shared_ptr<Value>& node : topo) {
    node->backward();
  }
}

std::vector<std::shared_ptr<Value>> Value::internalTopoSort() {
  std::unordered_set<std::shared_ptr<Value>> visited;
  std::vector<std::shared_ptr<Value>> topo;

  std::function<void(std::shared_ptr<Value>)> dfs =
      [&](const std::shared_ptr<Value>& v) {
        if (visited.find(v) == visited.end()) {
          visited.insert(v);
          for (const auto& child : v->prev) {
            dfs(child);
          }
          topo.push_back(v);
        }
      };

  dfs(shared_from_this());
  std::reverse(topo.begin(), topo.end());
  return topo;
}

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& a,
                                 const float b) {
  return a + Value::create(Value(static_cast<double>(b)));
}

std::shared_ptr<Value> operator+(const float b,
                                 const std::shared_ptr<Value>& a) {
  return Value::create(Value(static_cast<double>(b))) + a;
}

std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a,
                                 const float b) {
  return a - Value::create(Value(static_cast<double>(b)));
}

std::shared_ptr<Value> operator-(const float b,
                                 const std::shared_ptr<Value>& a) {
  return Value::create(Value(static_cast<double>(b))) - a;
}

std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& a,
                                 const float b) {
  return a * Value::create(Value(static_cast<double>(b)));
}

std::shared_ptr<Value> operator*(const float b,
                                 const std::shared_ptr<Value>& a) {
  return Value::create(Value(static_cast<double>(b))) * a;
}

std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& a,
                                 const float b) {
  return a / Value::create(Value(static_cast<double>(b)));
}

std::shared_ptr<Value> operator/(const float b,
                                 const std::shared_ptr<Value>& a) {
  return Value::create(Value(static_cast<double>(b))) / a;
}

std::ostream& operator<<(std::ostream& os, const Value& obj) {
  os << "Value(data=" << obj.data << ", grad=" << obj.grad << ", requires_grad=" << obj.requires_grad << ")";
  return os;
}
