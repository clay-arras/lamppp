#include "variable_pool.h"
#include <boost/pool/pool_alloc.hpp>
#include "test/benchmarks/mem_pool/variable.h"

boost::fast_pool_allocator<void> VariablePool::alloc_;

VariablePool::VariablePool() {
    impl_ = std::allocate_shared<VariableImpl>(alloc_, 0.0F);
}

VariablePool::VariablePool(float data) {
    impl_ = std::allocate_shared<VariableImpl>(alloc_, data);
}