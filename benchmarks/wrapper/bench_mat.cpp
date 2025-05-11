/* 04/09/25
-----------------------------------------------------------------------------
Benchmark                                   Time             CPU   Iterations
-----------------------------------------------------------------------------
BM_MatrixMultiplicationSharedValue 1553881315 ns   1553586661 ns            1
BM_MatrixMultiplicationVariable    2464257752 ns   2463430186 ns            1
BM_MatrixMultiplicationDouble        11449974 ns     11444813 ns           61
*/
#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include "variable.h"
#include "wrapper_engine.h"

namespace {

const int kRows1 = 784;
const int kCols1 = 256;
const int kCols2 = 10;

void BM_MatrixMultiplicationSharedValue(benchmark::State& state) {

  Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic> mat1(kRows1,
                                                                  kCols1);
  Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic> mat2(kCols1,
                                                                  kCols2);

  auto init_fn = [](const SharedValue&) {
    return SharedValue((2.0F * (static_cast<float>(rand()) / RAND_MAX) - 1.0F) /
                       1000);
  };

  mat1 = mat1.unaryExpr(
      [&init_fn](const SharedValue& val) { return init_fn(val); });
  mat2 = mat2.unaryExpr(
      [&init_fn](const SharedValue& val) { return init_fn(val); });

  for (auto _ : state) {
    Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic> res =
        mat1 * mat2;
    benchmark::DoNotOptimize(res);
  }
}

void BM_MatrixMultiplicationVariable(benchmark::State& state) {

  Eigen::Matrix<Variable, Eigen::Dynamic, Eigen::Dynamic> mat1(kRows1, kCols1);
  Eigen::Matrix<Variable, Eigen::Dynamic, Eigen::Dynamic> mat2(kCols1, kCols2);

  auto init_fn = [](const Variable&) {
    return Variable((2.0F * (static_cast<float>(rand()) / RAND_MAX) - 1.0F) /
                    1000);
  };

  mat1 =
      mat1.unaryExpr([&init_fn](const Variable& val) { return init_fn(val); });
  mat2 =
      mat2.unaryExpr([&init_fn](const Variable& val) { return init_fn(val); });

  for (auto _ : state) {
    Eigen::Matrix<Variable, Eigen::Dynamic, Eigen::Dynamic> res = mat1 * mat2;
    benchmark::DoNotOptimize(res);
  }
}

void BM_MatrixMultiplicationDouble(benchmark::State& state) {
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat1(kRows1, kCols1);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> mat2(kCols1, kCols2);

  auto init_fn = [](double) {
    return (2.0 * (static_cast<double>(rand()) / RAND_MAX) - 1.0) / 1000;
  };

  mat1 = mat1.unaryExpr([&init_fn](const double&) { return init_fn(0); });
  mat2 = mat2.unaryExpr([&init_fn](const double&) { return init_fn(0); });

  for (auto _ : state) {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> res = mat1 * mat2;
    benchmark::DoNotOptimize(res);
  }
}

BENCHMARK(BM_MatrixMultiplicationSharedValue);
BENCHMARK(BM_MatrixMultiplicationVariable);
BENCHMARK(BM_MatrixMultiplicationDouble);

}  // namespace

BENCHMARK_MAIN();
