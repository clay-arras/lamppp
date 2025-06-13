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

VARIABLE w/ REQUIRES_GRAD OPTIM

AddForward512     279040 ns       278830 ns         2339
SubForward512     267933 ns       267754 ns         2603
MulForward512     272223 ns       272072 ns         2584
DivForward512     229460 ns       229292 ns         2590
AbsForward512      25627 ns        25363 ns        27682
SinForward512      51497 ns        50793 ns        13896
CosForward512      50887 ns        50648 ns        13575

TENSOR

AddForward512     278111 ns       277921 ns         2380
SubForward512     268234 ns       268092 ns         2643
MulForward512     268476 ns       268329 ns         2609
DivForward512     229963 ns       229832 ns         2607
AbsForward512      25057 ns        24872 ns        27675
SinForward512      50264 ns        50076 ns        13978
CosForward512      50563 ns        50343 ns        13914








----------------------------
WITH TENSOR ASYNC
--------------------------------------------------------
Benchmark              Time             CPU   Iterations
--------------------------------------------------------
AddForward512      50081 ns        48190 ns        16867
SubForward512      58929 ns        56051 ns        13117
MulForward512      67239 ns        63719 ns        14748
DivForward512      62483 ns        60000 ns        10370
AbsForward512      40094 ns        38753 ns        19139
SinForward512      42026 ns        40517 ns        17120
CosForward512      48076 ns        46488 ns        19900


WITH VECTOR ASYNC
AddForward512      20180 ns        19460 ns        41849
SubForward512      19774 ns        19082 ns        42097
MulForward512      18620 ns        18014 ns        39648

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

  // std::vector<
  //     std::pair<std::string, std::function<Tensor(Tensor, Tensor)>>>
  //     bin_functions = {
  //         {"Add", add_Tensors},
  //         {"Sub", sub_Tensors},
  //         {"Mul", mul_Tensors},
  //         {"Div", div_Tensors},
  //     };
  // std::vector<std::pair<
  //     std::string,
  //     std::function<Tensor(
  //         Tensor)>>>  
  //     una_functions = {
  //         {"Abs", abs_Tensor},
  //         {"Sin", sin_Tensor},
  //         {"Cos", cos_Tensor},
  //     };

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
    {512, 512}
    //   {1024, 1024},
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
              cudaDeviceSynchronize();
              benchmark::DoNotOptimize(c);
            }
          });
    }
  }

  // for (const auto& pair : bin_functions) {
  //   const std::string& name = pair.first;
  //   const auto& fn = pair.second;
  //   for (const auto& shape : shapes) {
  //     benchmark::RegisterBenchmark(
  //         name + "Forward" + std::to_string(shape[0]),
  //         [fn, shape](benchmark::State& state) {
  //           for (auto _ : state) {
  //             state.PauseTiming();
  //             Tensor a =
  //                 rand(shape, DeviceType::CUDA, DataType::Float32, false).data();
  //             Tensor b =
  //                 rand(shape, DeviceType::CUDA, DataType::Float32, false).data();
  //             state.ResumeTiming();
  //             Tensor c = fn(a, b);
  //           }
  //         });
  //   }
  // }

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
              cudaDeviceSynchronize();
                                       benchmark::DoNotOptimize(c);
                                     }
                                   });
    }
  }

  // for (const auto& pair : una_functions) {
  //   const std::string& name = pair.first;
  //   const auto& fn = pair.second;
  //   for (const auto& shape : shapes) {
  //     benchmark::RegisterBenchmark(name + "Forward" + std::to_string(shape[0]),
  //                                  [fn, shape](benchmark::State& state) {
  //                                    for (auto _ : state) {
  //                                      state.PauseTiming();
  //                                      Tensor a =
  //                                          rand(shape, DeviceType::CUDA,
  //                                               DataType::Float32, false).data();
  //                                      state.ResumeTiming();
  //                                      Tensor c = fn(a);
  //                                    }
  //                                  });
  //   }
  // }

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