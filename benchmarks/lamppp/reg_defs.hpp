#pragma once

#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>
#include <array>
#include <functional>
#include <string>
#include "lamppp/lamppp.hpp"

template <size_t N>
using OperatorFunction = std::function<lmp::autograd::Variable(
    const std::array<lmp::autograd::Variable, N>&)>;
template <size_t N>
using InitializerFunction =
    std::function<std::array<lmp::autograd::Variable, N>(bool)>;

const size_t kIterations = 10;
const float kWarmUpTime = 1;

template <size_t N>
void register_forward(const std::string& name, OperatorFunction<N> op_fn,
                      InitializerFunction<N> init_fn) {
  benchmark::RegisterBenchmark(
      name + "Forward",
      [op_fn, init_fn](benchmark::State& state) {
        for (auto _ : state) {
          state.PauseTiming();
          std::array<lmp::autograd::Variable, N> inputs = init_fn(false);
          state.ResumeTiming();
          lmp::autograd::Variable result = op_fn(inputs);
          benchmark::DoNotOptimize(result);
        }
      })
      ->MinWarmUpTime(kWarmUpTime)
      ->Teardown([](const benchmark::State& state) {
        cudaStreamSynchronize(0);
        cudaDeviceSynchronize();
      });
}

template <size_t N>
void register_backward(const std::string& name, OperatorFunction<N> op_fn,
                       InitializerFunction<N> init_fn) {
  benchmark::RegisterBenchmark(
      name + "Backward",
      [op_fn, init_fn](benchmark::State& state) {
        for (auto _ : state) {
          state.PauseTiming();
          std::array<lmp::autograd::Variable, N> inputs = init_fn(true);
          lmp::autograd::Variable result = op_fn(inputs);
          state.ResumeTiming();
          result.backward();
          benchmark::DoNotOptimize(result);
        }
      })
      ->MinWarmUpTime(kWarmUpTime)
      ->Teardown([](const benchmark::State& state) {
        cudaStreamSynchronize(0);
        cudaDeviceSynchronize();
      });
}