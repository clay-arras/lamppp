#include "tensor.hpp"
#include <cassert>
#include <iostream>

namespace autograd {

std::ostream& operator<<(std::ostream& os, const Tensor& obj) {
  obj.impl_->print_(os);
  return os;
}

}  // namespace autograd