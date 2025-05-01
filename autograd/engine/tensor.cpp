#include "tensor.hpp"
#include <cassert>
#include <iostream>

namespace autograd {

std::ostream& operator<<(std::ostream& os, const Tensor& obj) {
  os << "Tensor(impl=" << *obj.impl_;
  os << ")";
  return os;
}

}  // namespace autograd