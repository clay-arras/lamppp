#include <benchmark/benchmark.h>
#include <memory>
#include <vector>
#include "op_defs.hpp"

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);

  std::vector<lmp::tensor::DataType> dtypes = {lmp::tensor::DataType::Float32};
  std::vector<lmp::tensor::DeviceType> devices = {
      lmp::tensor::DeviceType::CPU, lmp::tensor::DeviceType::CUDA};
  std::vector<std::vector<size_t>> sqr_shapes = {
      {512, 512},
  };
  std::vector<std::vector<size_t>> rect_shapes = {
      {64, 512},
  };
  std::vector<std::array<std::vector<size_t>, 2>> cast_shapes = {
      {{{1, 64, 1}, {64, 1, 64}}}};

  std::vector<std::unique_ptr<BinaryOperatorBase>> binary_operators;
  binary_operators.push_back(std::make_unique<AddOp>());
  binary_operators.push_back(std::make_unique<SubOp>());
  binary_operators.push_back(std::make_unique<MulOp>());
  binary_operators.push_back(std::make_unique<DivOp>());

  for (const auto& op : binary_operators) {
    for (const auto& dtype : dtypes) {
      for (const auto& device : devices) {
        for (const auto& shape : sqr_shapes) {
          OperatorConfig<2> config{
              .shapes = {shape, shape}, .device = device, .dtype = dtype};
          op->register_benchmarks(config);
        }
        for (const auto& shapes : cast_shapes) {
          OperatorConfig<2> config{
              .shapes = shapes, .device = device, .dtype = dtype};
          op->register_benchmarks(config);
        }
      }
    }
  }

  std::vector<std::unique_ptr<UnaryOperatorBase>> unary_operators;
  unary_operators.push_back(std::make_unique<NegOp>());
  unary_operators.push_back(std::make_unique<LogOp>());
  unary_operators.push_back(std::make_unique<ExpOp>());
  unary_operators.push_back(std::make_unique<SqrtOp>());
  unary_operators.push_back(std::make_unique<AbsOp>());
  unary_operators.push_back(std::make_unique<SinOp>());
  unary_operators.push_back(std::make_unique<CosOp>());
  unary_operators.push_back(std::make_unique<TanOp>());
  unary_operators.push_back(std::make_unique<ClampOp>());

  for (const auto& op : unary_operators) {
    for (const auto& dtype : dtypes) {
      for (const auto& device : devices) {
        for (const auto& shape : sqr_shapes) {
          OperatorConfig<1> config{
              .shapes = {shape}, .device = device, .dtype = dtype};
          op->register_benchmarks(config);
        }
      }
    }
  }

  std::vector<std::unique_ptr<ReductOperatorBase>> reduct_operators;
  reduct_operators.push_back(std::make_unique<SumOp>());
  reduct_operators.push_back(std::make_unique<MinOp>());
  reduct_operators.push_back(std::make_unique<MaxOp>());
  reduct_operators.push_back(std::make_unique<ProdOp>());

  for (const auto& op : reduct_operators) {
    for (const auto& dtype : dtypes) {
      for (const auto& device : devices) {
        for (const auto& shape : rect_shapes) {
          OperatorConfig<1> config{
              .shapes = {shape}, .device = device, .dtype = dtype};
          op->register_benchmarks(config);
        }
      }
    }
  }

  benchmark::RunSpecifiedBenchmarks();
  return 0;
}