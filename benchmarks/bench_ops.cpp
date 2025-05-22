#include <benchmark/benchmark.h>
#include <functional>
#include <string>
#include <vector>
#include "lamppp/autograd/functions/unary_ops.hpp"
#include "lamppp/lamppp.hpp"

using lmp::autograd::rand;
using lmp::autograd::Variable;
using lmp::tensor::DataType;
using lmp::tensor::DeviceType;

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
Variable abs_variable(const Variable& a) {
  return lmp::autograd::ops::abs(a);
}
Variable sin_variable(const Variable& a) {
  return lmp::autograd::ops::sin(a);
}
Variable cos_variable(const Variable& a) {
  return lmp::autograd::ops::cos(a);
}
}  // anonymous namespace

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);

  std::vector<
      std::pair<std::string, std::function<Variable(Variable, Variable)>>>
      bin_functions = {
          {"Add", add_variables},
          {"Sub", sub_variables},
          {"Mul", mul_variables},
          {"Div", div_variables},
      };
  std::vector<std::pair<
      std::string,
      std::function<Variable(
          Variable)>>>  // TODO: I'm too lazy to do the other ones because of domain constraints. do it later
      una_functions = {
          {"Abs", abs_variable},
          {"Sin", sin_variable},
          {"Cos", cos_variable},
      };
  std::vector<std::vector<size_t>> shapes = {
      {128, 128},
      {256, 256},
      {1024, 1024},
  };

  for (const auto& pair : bin_functions) {
    const std::string& name = pair.first;
    const auto& fn = pair.second;
    for (const auto& shape : shapes) {
      benchmark::RegisterBenchmark(
          name + "Forward" + std::to_string(shape[0]),
          [fn, shape](benchmark::State& state) {
            for (auto _ : state) {
              state.PauseTiming();
              Variable a =
                  rand(shape, DeviceType::CUDA, DataType::Float32, false);
              Variable b =
                  rand(shape, DeviceType::CUDA, DataType::Float32, false);
              state.ResumeTiming();
              Variable c = fn(a, b);
            }
          });
    }
  }

  for (const auto& pair : bin_functions) {
    const std::string& name = pair.first;
    const auto& fn = pair.second;
    for (const auto& shape : shapes) {
      benchmark::RegisterBenchmark(
          name + "Backward" + std::to_string(shape[0]),
          [fn, shape](benchmark::State& state) {
            for (auto _ : state) {
              state.PauseTiming();
              Variable a =
                  rand(shape, DeviceType::CUDA, DataType::Float32, true);
              Variable b =
                  rand(shape, DeviceType::CUDA, DataType::Float32, true);
              Variable c = fn(a, b);
              state.ResumeTiming();
              c.backward();
            }
          });
    }
  }

  for (const auto& pair : una_functions) {
    const std::string& name = pair.first;
    const auto& fn = pair.second;
    for (const auto& shape : shapes) {
      benchmark::RegisterBenchmark(name + "Forward" + std::to_string(shape[0]),
                                   [fn, shape](benchmark::State& state) {
                                     for (auto _ : state) {
                                       state.PauseTiming();
                                       Variable a =
                                           rand(shape, DeviceType::CUDA,
                                                DataType::Float32, false);
                                       state.ResumeTiming();
                                       Variable c = fn(a);
                                     }
                                   });
    }
  }

  for (const auto& pair : una_functions) {
    const std::string& name = pair.first;
    const auto& fn = pair.second;
    for (const auto& shape : shapes) {
      benchmark::RegisterBenchmark(name + "Backward" + std::to_string(shape[0]),
                                   [fn, shape](benchmark::State& state) {
                                     for (auto _ : state) {
                                       state.PauseTiming();
                                       Variable a =
                                           rand(shape, DeviceType::CUDA,
                                                DataType::Float32, true);
                                       Variable c = fn(a);
                                       state.ResumeTiming();
                                       c.backward();
                                     }
                                   });
    }
  }

  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
