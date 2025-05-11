#include <benchmark/benchmark.h>
#include <functional>
#include <string>
#include <vector>
#include "include/lamppp/tensor/constructor.hpp"
#include "include/lamppp/tensor/data_type.hpp"
#include "include/lamppp/tensor/device_type.hpp"
#include "include/lamppp/tensor/variable.hpp"
#include "include/lamppp/tensor/variable_ops.hpp"

using autograd::rand;
using autograd::Variable;

namespace {
Variable add_variables(const Variable& a, const Variable& b) {
  return a + b;
}
Variable sub_variables(const Variable& a, const Variable& b) {
  return a - b;
}
Variable mul_variables(const Variable& a, const Variable& b) {
  return a * b;
}
Variable div_variables(const Variable& a, const Variable& b) {
  return a / b;
}
}  // anonymous namespace

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);

  std::vector<
      std::pair<std::string, std::function<Variable(Variable, Variable)>>>
      functions = {
          {"Add", add_variables},
          {"Sub", sub_variables},
          {"Mul", mul_variables},
          {"Div", div_variables},
      };
  std::vector<std::vector<size_t>> shapes = {
      {128, 128},
      {256, 256},
      {1024, 1024},
  };

  for (const auto& pair : functions) {
    const std::string& name = pair.first;
    const auto& fn = pair.second;

    for (std::vector<size_t> shape : shapes) {
      benchmark::RegisterBenchmark(name, [fn, &shape](benchmark::State& state) {
        for (auto _ : state) {
          state.PauseTiming();
          Variable a = rand(shape, DeviceType::CUDA, DataType::Float32, false);
          Variable b = rand(shape, DeviceType::CUDA, DataType::Float32, false);
          state.ResumeTiming();

          Variable c = fn(a, b);
        }
      });
    }
  }

  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
