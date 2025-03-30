#include "engine.h"

Value::Value(double data, std::unordered_set<std::shared_ptr<Value>> children,
             double grad)
    : data(data), grad(grad), prev(children), 
      backward_fn(nullptr), backward_ctx(nullptr) {}

std::shared_ptr<Value> operator+(const std::shared_ptr<Value> &a,
                                 const std::shared_ptr<Value> &b) {
  return a->operator+(b);
}

std::shared_ptr<Value> Value::operator+(const std::shared_ptr<Value> &other) {
  std::shared_ptr<Value> out = std::make_shared<Value>(
      this->data + other->data,
      std::unordered_set<std::shared_ptr<Value>>{shared_from_this(), other});
  
  auto ctx = std::make_shared<AddBackwardContext>(shared_from_this(), other, out);
  out->backward_fn = &add_backward;
  out->backward_ctx = static_cast<void*>(ctx);
  
  return out;
}

std::shared_ptr<Value> operator-(const std::shared_ptr<Value> &a,
                                 const std::shared_ptr<Value> &b) {
  return a->operator-(b);
}

std::shared_ptr<Value> Value::operator-(const std::shared_ptr<Value> &other) {
  return shared_from_this() + (other * std::make_shared<Value>(-1));
}

std::shared_ptr<Value> operator*(const std::shared_ptr<Value> &a,
                                 const std::shared_ptr<Value> &b) {
  return a->operator*(b);
}

std::shared_ptr<Value> Value::operator*(const std::shared_ptr<Value> &other) {
  std::shared_ptr<Value> out = std::make_shared<Value>(
      this->data * other->data,
      std::unordered_set<std::shared_ptr<Value>>{shared_from_this(), other});
  
  auto ctx = std::make_shared<MulBackwardContext>(shared_from_this(), other, out);
  out->backward_fn = &mul_backward;
  out->backward_ctx = static_cast<void*>(ctx);
  
  return out;
}

std::shared_ptr<Value> operator/(const std::shared_ptr<Value> &a,
                                 const std::shared_ptr<Value> &b) {
  return a->operator/(b);
}

std::shared_ptr<Value> Value::operator/(const std::shared_ptr<Value> &other) {
  return shared_from_this() * other->pow(std::make_shared<Value>(-1));
}

std::shared_ptr<Value> Value::pow(std::shared_ptr<Value> other) {
  std::shared_ptr<Value> out = std::make_shared<Value>(
      std::pow(this->data, other->data),
      std::unordered_set<std::shared_ptr<Value>>{shared_from_this(), other});
  
  auto ctx = std::make_shared<PowBackwardContext>(shared_from_this(), other, out);
  out->backward_fn = &pow_backward;
  out->backward_ctx = static_cast<void*>(ctx);
  
  return out;
}

std::shared_ptr<Value> Value::exp() {
  std::shared_ptr<Value> out = std::make_shared<Value>(
      std::exp(this->data),
      std::unordered_set<std::shared_ptr<Value>>{shared_from_this()});
  
  auto ctx = std::make_shared<ExpBackwardContext>(shared_from_this(), out);
  out->backward_fn = &exp_backward;
  out->backward_ctx = static_cast<void*>(ctx);
  
  return out;
}

std::shared_ptr<Value> exp(const std::shared_ptr<Value> &a) {
  return a->exp();
}

std::shared_ptr<Value> Value::log() {
  assert(this->data > 0);
  std::shared_ptr<Value> out = std::make_shared<Value>(
      std::log(this->data),
      std::unordered_set<std::shared_ptr<Value>>{shared_from_this()});
  
  auto ctx = std::make_shared<LogBackwardContext>(shared_from_this(), out);
  out->backward_fn = &log_backward;
  out->backward_ctx = static_cast<void*>(ctx);
  
  return out;
}

std::shared_ptr<Value> log(const std::shared_ptr<Value> &a) {
  return a->log();
}

// TODO: make normal
std::shared_ptr<Value> Value::tanh() {
  std::shared_ptr<Value> exp =
      (std::make_shared<Value>(2) * shared_from_this())->exp();
  std::shared_ptr<Value> one = std::make_shared<Value>(1);
  return (exp - one) / (exp + one);
}

std::shared_ptr<Value> tanh(const std::shared_ptr<Value> &a) {
  return a->tanh();
}

std::shared_ptr<Value> Value::relu() {
  double out_data = std::max(0.0, this->data);
  std::shared_ptr<Value> out = std::make_shared<Value>(
      out_data, std::unordered_set<std::shared_ptr<Value>>{shared_from_this()});

  auto ctx = std::make_shared<ReluBackwardContext>(shared_from_this(), out);
  out->backward_fn = &relu_backward;
  out->backward_ctx = static_cast<void*>(ctx);
  
  return out;
}

std::shared_ptr<Value> relu(const std::shared_ptr<Value> &a) {
  return a->relu();
}

void Value::backprop() {
  std::vector<std::shared_ptr<Value>> topo = internalTopoSort();
  this->grad = 1.0;
  for (std::shared_ptr<Value> node : topo) {
    node->backward();
  }
}

std::vector<std::shared_ptr<Value>> Value::internalTopoSort() {
  std::unordered_set<std::shared_ptr<Value>> visited;
  std::vector<std::shared_ptr<Value>> topo;

  std::function<void(std::shared_ptr<Value>)> dfs =
      [&](std::shared_ptr<Value> v) {
        if (visited.find(v) == visited.end()) {
          visited.insert(v);
          for (const auto &child : v->prev) {
            dfs(child);
          }
          topo.push_back(v);
        }
      };

  dfs(shared_from_this());
  std::reverse(topo.begin(), topo.end());
  return topo;
}

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& a, const float b) {
  return a + std::make_shared<Value>(b);
}

std::shared_ptr<Value> operator+(const float b, const std::shared_ptr<Value>& a) {
  return std::make_shared<Value>(b) + a;
}

std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a, const float b) {
  return a - std::make_shared<Value>(b);
}

std::shared_ptr<Value> operator-(const float b, const std::shared_ptr<Value>& a) {
  return std::make_shared<Value>(b) - a;
}

std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& a, const float b) {
  return a * std::make_shared<Value>(b);
}

std::shared_ptr<Value> operator*(const float b, const std::shared_ptr<Value>& a) {
  return std::make_shared<Value>(b) * a;
}

std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& a, const float b) {
  return a / std::make_shared<Value>(b);
}

std::shared_ptr<Value> operator/(const float b, const std::shared_ptr<Value>& a) {
  return std::make_shared<Value>(b) / a;
}

std::ostream &operator<<(std::ostream &os, const Value &obj) {
  os << "Value(data=" << obj.data << ", grad=" << obj.grad << ")";
  return os;
}