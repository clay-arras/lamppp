#include "value_pool.h"
#include <cstddef>
#include <mutex>
#include "engine.h"

thread_local std::vector<void*> ValueMemoryPool::local_pool_;

ValueMemoryPool::ValueMemoryPool(size_t size) {
    block_size_ = sizeof(Value);
    initial_local_size_ = size / 4;
    local_pool_.reserve(initial_local_size_); // heuristic
    for (size_t i = 0; i < initial_local_size_; i++) {
        local_pool_.push_back(::operator new(block_size_));
    }
}

void* ValueMemoryPool::allocate() {
    if (local_pool_.empty()) {
        initialize_local_pool();
    }

    if (!local_pool_.empty()) {
        void* alloc = local_pool_.back();
        local_pool_.pop_back();
        return alloc;
    } 
    
    {
        assert(false);
        std::lock_guard<std::mutex> lock(global_mutex_);
        if (!global_pool_.empty()) {
            void* alloc = global_pool_.back();
            global_pool_.pop_back();
            return alloc;
        }
    }
    
    assert(false);
    return ::operator new(block_size_);
}

void ValueMemoryPool::deallocate(void* value) { 
    local_pool_.push_back(value);
    
    if (local_pool_.size() > 1000) { // TODO(nlin): do resize better
        std::lock_guard<std::mutex> lock(global_mutex_);
        for (size_t i = 0; i < 500; i++) {
            global_pool_.push_back(local_pool_.back());
            local_pool_.pop_back();
        }
    }
}
