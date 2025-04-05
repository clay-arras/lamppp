#include "value_pool.h"
#include <cstddef>
#include <mutex>
#include "engine.h"


ValueMemoryPool::ValueMemoryPool(size_t size) {
    pool_.reserve(size);
    for (size_t i = 0; i < size; i++) {
        pool_.push_back(::operator new(sizeof(Value)));
    }
}

void* ValueMemoryPool::allocate() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pool_.empty()) {
        assert(false); // TODO(nlin): just crash for now, later implement resizing like std::vector
    }
    void* alloc = pool_.back();
    pool_.pop_back();
    return alloc; 
}

void ValueMemoryPool::deallocate(void* value) { 
    std::lock_guard<std::mutex> lock(mutex_);
    pool_.push_back(value); // can't check void*, if this isn't a Value it's going to fucking die.
}
