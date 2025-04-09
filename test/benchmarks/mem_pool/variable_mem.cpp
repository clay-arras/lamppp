#include "variable_mem.h"
#include "value_pool.h"

ValueMemoryPool VariableMem::pool_(100000, sizeof(VariableImpl));

void VariableMem::destroy(VariableImpl* ptr) {
    ptr->~VariableImpl();  
    VariableMem::pool_.deallocate(ptr);
}

VariableMem::VariableMem() {
    void* raw_memory = pool_.allocate();
    auto* impl = new (raw_memory) VariableImpl(0.0F);
    impl_ = std::shared_ptr<VariableImpl>(impl, destroy);
}

VariableMem::VariableMem(float data) {
    void* raw_memory = pool_.allocate();
    auto* impl = new (raw_memory) VariableImpl(data);
    impl_ = std::shared_ptr<VariableImpl>(impl, destroy);
}