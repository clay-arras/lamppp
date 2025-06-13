#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include "lamppp/lamppp.hpp"
#include "lamppp/tensor/fill_like.hpp"

namespace {

void BM_Variable_Grad(benchmark::State& state) {
    std::vector<size_t> shape = {512, 512};
    
    for (auto _ : state) {
        state.PauseTiming();
        lmp::Variable a =
            lmp::autograd::randn(0, 1, shape, lmp::DeviceType::CUDA, lmp::DataType::Float32, true);
        lmp::Variable b =
            lmp::autograd::randn(0, 1, shape, lmp::DeviceType::CUDA, lmp::DataType::Float32, true);
        state.ResumeTiming();
        lmp::Variable c = a + b;
    }
}
void BM_Variable_NoGrad(benchmark::State& state) {
    std::vector<size_t> shape = {512, 512};
    
    for (auto _ : state) {
        state.PauseTiming();
        lmp::Variable a =
            lmp::autograd::randn(0, 1, shape, lmp::DeviceType::CUDA, lmp::DataType::Float32, false);
        lmp::Variable b =
            lmp::autograd::randn(0, 1, shape, lmp::DeviceType::CUDA, lmp::DataType::Float32, false);
        state.ResumeTiming();
        lmp::Variable c = a + b;
    }
}
void BM_Tensor(benchmark::State& state) {
    std::vector<size_t> shape = {512, 512};
    
    for (auto _ : state) {
        state.PauseTiming();
        lmp::Tensor a =
            lmp::autograd::randn(0, 1, shape, lmp::DeviceType::CUDA, lmp::DataType::Float32, false).data();
        lmp::Tensor b =
            lmp::autograd::randn(0, 1, shape, lmp::DeviceType::CUDA, lmp::DataType::Float32, false).data();
        state.ResumeTiming();
        lmp::Tensor c = a + b;
    }
}
void BM_TensorOnesLike(benchmark::State& state) {
    std::vector<size_t> shape = {512, 512};
    lmp::Tensor a = lmp::autograd::randn(0, 1, shape, lmp::DeviceType::CUDA, lmp::DataType::Float32, false).data();
    
    for (auto _ : state) {
        lmp::Tensor c = lmp::tensor::ones_like(a);
    }
}

BENCHMARK(BM_Variable_Grad)->Iterations(1000);
BENCHMARK(BM_Variable_NoGrad)->Iterations(1000);
BENCHMARK(BM_Tensor)->Iterations(1000);
BENCHMARK(BM_TensorOnesLike)->Iterations(1000);

}


BENCHMARK_MAIN();
