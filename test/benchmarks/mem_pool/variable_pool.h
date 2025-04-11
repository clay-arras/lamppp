#ifndef VARIABLE_POOL_H
#define VARIABLE_POOL_H

#include <boost/pool/pool_alloc.hpp>
#include "test/benchmarks/mem_pool/variable.h"

class VariablePool : public Variable {
 public:
  VariablePool();
  explicit VariablePool(float data);

  explicit VariablePool(std::shared_ptr<VariableImpl>& impl) : Variable(impl) {}

  static boost::fast_pool_allocator<void> alloc_;
};

#endif  //VARIABLE_POOL_H
