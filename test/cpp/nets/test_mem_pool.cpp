#include <cassert>
#include <iostream>
#include <memory>
#include <vector>
#include "autograd/engine/engine.h"
#include "test/benchmarks/mem_pool/value_pool.h"

namespace {
void test_basic_allocation_deallocation() {
  ValueMemoryPool pool(10, sizeof(Value));
  void* ptr = pool.allocate();
  assert(ptr != nullptr);

  pool.deallocate(ptr);
  std::cout << "Basic allocation and deallocation test passed!" << std::endl;
}

void test_value_creation() {
  Value val(5.0, true);
  std::shared_ptr<Value> v = Value::create(val);

  assert(v->data == 5.0);
  assert(v->requires_grad == true);
  assert(v->grad == 0.0);
  std::cout << "Value creation test passed!" << std::endl;
}

void test_multiple_allocations() {
  ValueMemoryPool pool(5, sizeof(Value));
  std::vector<void*> ptrs;

  for (int i = 0; i < 5; i++) {
    void* ptr = pool.allocate();
    assert(ptr != nullptr);
    ptrs.push_back(ptr);
  }

  for (void* ptr : ptrs) {
    pool.deallocate(ptr);
  }

  void* ptr = pool.allocate();
  assert(ptr != nullptr);
  std::cout << "Multiple allocations test passed!" << std::endl;
}
}  // namespace

int main() {
  test_basic_allocation_deallocation();
  test_value_creation();
  test_multiple_allocations();

  std::cout << "All memory pool tests passed!" << std::endl;
  return 0;
}
