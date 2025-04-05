#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include "autograd/engine/wrapper_engine.h"
#include "test/cpp/dummy_value.h"
// #include "autograd/util/value_eigen_support.h"

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


void BM_MatrixMultiplicationFloat(benchmark::State& state) {
  Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic> mat1(kRows1,
                                                              kCols1);
  Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic> mat2(kCols1,
                                                              kCols2);

  auto init_fn = [](float) {
    return Float(
        (2.0F * (static_cast<float>(rand()) / RAND_MAX) - 1.0F) / 1000);
  };

  mat1 = mat1.unaryExpr([&init_fn](const Float&) { return init_fn(0); });
  mat2 = mat2.unaryExpr([&init_fn](const Float&) { return init_fn(0); });

  for (auto _ : state) {
    Eigen::Matrix<Float, Eigen::Dynamic, Eigen::Dynamic> res =
        mat1 * mat2;
    benchmark::DoNotOptimize(res);
  }
}

BENCHMARK(BM_MatrixMultiplicationSharedValue);
BENCHMARK(BM_MatrixMultiplicationFloat);

}  // namespace

BENCHMARK_MAIN();
