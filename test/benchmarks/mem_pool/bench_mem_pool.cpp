/*
Run on (32 X 1461.43 MHz CPU s)
CPU Caches:
  L1 Data 48 KiB (x16)
  L1 Instruction 32 KiB (x16)
  L2 Unified 2048 KiB (x16)
  L3 Unified 36864 KiB (x1)
Load Average: 0.98, 1.44, 1.03
----------------------------------------------------------
Benchmark                Time             CPU   Iterations
----------------------------------------------------------
BM_IntMemPool         88.8 ns         88.8 ns      8379708
BM_IntPoolAlloc        168 ns          168 ns      4193977
BM_IntNoMemPool       58.5 ns         58.5 ns     11979298
BM_VarMemPool          202 ns          202 ns      3501338
BM_VarPoolAlloc        285 ns          285 ns      2410470
BM_VarNoMemPool       75.4 ns         75.4 ns     10089873
*/

#include <benchmark/benchmark.h>
#include <boost/pool/pool_alloc.hpp>
#include <memory>
#include "value_pool.h"
#include "variable.h"
#include "variable_mem.h"
#include "variable_pool.h"
#include <boost/pool/poolfwd.hpp>

namespace {

ValueMemoryPool int_pool(10000, sizeof(int));
boost::fast_pool_allocator<void> alloc;

void deleter(int* ptr) {
    int_pool.deallocate(static_cast<void*>(ptr));
}

void BM_IntMemPool(benchmark::State &state) {
    for (auto _ : state) {
        void* raw_memory = static_cast<int*>(int_pool.allocate());
        int* block = new (raw_memory) int(-1);
        std::shared_ptr<int> ptr(block, &deleter);
        benchmark::DoNotOptimize(ptr);
    }
}

void BM_IntPoolAlloc(benchmark::State &state) {
  for (auto _ : state) {
    auto ptr = std::allocate_shared<int>(alloc, -1);
    benchmark::DoNotOptimize(ptr);
  }
}

void BM_IntNoMemPool(benchmark::State &state) {
    for (auto _ : state) {
        auto ptr = std::make_shared<int>(-1);
        benchmark::DoNotOptimize(ptr);
    }
}

BENCHMARK(BM_IntMemPool);
BENCHMARK(BM_IntPoolAlloc);
BENCHMARK(BM_IntNoMemPool);

void BM_VarMemPool(benchmark::State &state) {
    for (auto _ : state) {
        VariableMem var(-1.0F);
        benchmark::DoNotOptimize(var);
    }
}

void BM_VarPoolAlloc(benchmark::State &state) {
  for (auto _ : state) {
    VariablePool var(-1.0F);
    benchmark::DoNotOptimize(var);
  }
}

void BM_VarNoMemPool(benchmark::State &state) {
    for (auto _ : state) {
        Variable var(-1.0F);
        benchmark::DoNotOptimize(var);
    }
}

BENCHMARK(BM_VarMemPool);
BENCHMARK(BM_VarPoolAlloc);
BENCHMARK(BM_VarNoMemPool);

}

BENCHMARK_MAIN();