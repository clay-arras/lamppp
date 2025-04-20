#include "autograd/engine/variable.h"
#include "autograd/engine/constructor.h" 
#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <functional>

using autograd::Variable;
using autograd::rand;

namespace {
Variable add_variables(const Variable& a, const Variable& b) { return a + b; }
Variable sub_variables(const Variable& a, const Variable& b) { return a - b; }
Variable mul_variables(const Variable& a, const Variable& b) { return a * b; }
Variable div_variables(const Variable& a, const Variable& b) { return a / b; }
}  // anonymous namespace

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);

  std::vector<std::pair<std::string, std::function<Variable(Variable, Variable)>>> functions = {
    {"Add", add_variables},
    {"Sub", sub_variables},
    {"Mul", mul_variables},
    {"Div", div_variables},
  };
  std::vector<std::vector<int>> shapes = {
    {128, 128},
    {256, 256},
    {1024, 1024},
  };

  for (const auto& pair : functions) {
    const std::string& name = pair.first;
    const auto& fn = pair.second;
    
    for (auto &shape : shapes) {
      benchmark::RegisterBenchmark(
        name,
        [fn, &shape](benchmark::State& state) {
          for (auto _ : state) {
            state.PauseTiming();
            Variable a = rand(shape);
            Variable b = rand(shape);
            state.ResumeTiming();

            Variable c = fn(a, b);
          }
        }
      );
    }
  }

  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
