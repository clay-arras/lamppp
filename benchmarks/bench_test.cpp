#include <benchmark/benchmark.h>
#include <functional>
#include <string>
#include <vector>
#include "lamppp/autograd/functions/unary_ops.hpp"
#include "lamppp/lamppp.hpp"

/*

VARIABLE

AddForward512    1744182 ns      1707565 ns          445
SubForward512    2431078 ns      2423377 ns          268
MulForward512    2404861 ns      2397370 ns          292
DivForward512    2389836 ns      2387234 ns          291
AbsForward512    2409429 ns      2145163 ns          332
SinForward512    2118312 ns      2111205 ns          333
CosForward512    2135547 ns      2128832 ns          329

TENSOR 

AddForward512      74200 ns        73845 ns         9672
SubForward512      75825 ns        75556 ns         9363
MulForward512      81487 ns        81125 ns         9012
DivForward512      84593 ns        84265 ns         8790
AbsForward512     215472 ns       215197 ns         3272
SinForward512      53431 ns        53101 ns        13247
CosForward512      53591 ns        53328 ns        13178

*/

using lmp::autograd::rand;
using lmp::autograd::Variable;
using lmp::tensor::Tensor;
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
Tensor add_Tensors(const Tensor& a, const Tensor& b) {
  return a + b;
}
Tensor sub_Tensors(const Tensor& a, const Tensor& b) {
  return a - b;
}
Tensor mul_Tensors(const Tensor& a, const Tensor& b) {
  return a * b;
}
Tensor div_Tensors(const Tensor& a, const Tensor& b) {
  return a / b;
}
Tensor abs_Tensor(const Tensor& a) {
  return lmp::tensor::ops::abs(a);
}
Tensor sin_Tensor(const Tensor& a) {
  return lmp::tensor::ops::sin(a);
}
Tensor cos_Tensor(const Tensor& a) {
  return lmp::tensor::ops::cos(a);
}
}  // anonymous namespace

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);

//   std::vector<
//       std::pair<std::string, std::function<Tensor(Tensor, Tensor)>>>
//       bin_functions = {
//           {"Add", add_Tensors},
//           {"Sub", sub_Tensors},
//           {"Mul", mul_Tensors},
//           {"Div", div_Tensors},
//       };
//   std::vector<std::pair<
//       std::string,
//       std::function<Tensor(
//           Tensor)>>>  
//       una_functions = {
//           {"Abs", abs_Tensor},
//           {"Sin", sin_Tensor},
//           {"Cos", cos_Tensor},
//       };
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
          Variable)>>>  
      una_functions = {
          {"Abs", abs_variable},
          {"Sin", sin_variable},
          {"Cos", cos_variable},
      };
  std::vector<std::vector<size_t>> shapes = {
    //   {128, 128},
    //   {256, 256},
    //   {1024, 1024},
    {512, 512}
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

//   for (const auto& pair : bin_functions) {
//     const std::string& name = pair.first;
//     const auto& fn = pair.second;
//     for (const auto& shape : shapes) {
//       benchmark::RegisterBenchmark(
//           name + "Forward" + std::to_string(shape[0]),
//           [fn, shape](benchmark::State& state) {
//             for (auto _ : state) {
//               state.PauseTiming();
//               Tensor a =
//                   rand(shape, DeviceType::CUDA, DataType::Float32, false).data();
//               Tensor b =
//                   rand(shape, DeviceType::CUDA, DataType::Float32, false).data();
//               state.ResumeTiming();
//               Tensor c = fn(a, b);
//             }
//           });
//     }
//   }

//   for (const auto& pair : bin_functions) {
//     const std::string& name = pair.first;
//     const auto& fn = pair.second;
//     for (const auto& shape : shapes) {
//       benchmark::RegisterBenchmark(
//           name + "Backward" + std::to_string(shape[0]),
//           [fn, shape](benchmark::State& state) {
//             for (auto _ : state) {
//               state.PauseTiming();
//               Variable a =
//                   rand(shape, DeviceType::CUDA, DataType::Float32, true);
//               Variable b =
//                   rand(shape, DeviceType::CUDA, DataType::Float32, true);
//               Variable c = fn(a, b);
//               state.ResumeTiming();
//               c.backward();
//             }
//           });
//     }
//   }
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

//   for (const auto& pair : una_functions) {
//     const std::string& name = pair.first;
//     const auto& fn = pair.second;
//     for (const auto& shape : shapes) {
//       benchmark::RegisterBenchmark(name + "Forward" + std::to_string(shape[0]),
//                                    [fn, shape](benchmark::State& state) {
//                                      for (auto _ : state) {
//                                        state.PauseTiming();
//                                        Tensor a =
//                                            rand(shape, DeviceType::CUDA,
//                                                 DataType::Float32, false).data();
//                                        state.ResumeTiming();
//                                        Tensor c = fn(a);
//                                      }
//                                    });
//     }
//   }

//   for (const auto& pair : una_functions) {
//     const std::string& name = pair.first;
//     const auto& fn = pair.second;
//     for (const auto& shape : shapes) {
//       benchmark::RegisterBenchmark(name + "Backward" + std::to_string(shape[0]),
//                                    [fn, shape](benchmark::State& state) {
//                                      for (auto _ : state) {
//                                        state.PauseTiming();
//                                        Variable a =
//                                            rand(shape, DeviceType::CUDA,
//                                                 DataType::Float32, true);
//                                        Variable c = fn(a);
//                                        state.ResumeTiming();
//                                        c.backward();
//                                      }
//                                    });
//     }
//   }

  benchmark::RunSpecifiedBenchmarks();
  return 0;
}