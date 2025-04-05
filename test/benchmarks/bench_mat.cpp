#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include "autograd/engine/wrapper_engine.h"

namespace {

  const int kRows1 = 784;
  const int kCols1 = 256;
  const int kCols2 = 10;

void BM_MatrixMultiplicationSharedValue(benchmark::State& state) {
  
  Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic> mat1(kRows1, kCols1);
  Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic> mat2(kCols1, kCols2);

  auto init_fn = [](const SharedValue&) {
    return SharedValue((2.0F * (static_cast<float>(rand()) / RAND_MAX) - 1.0F) / 1000);
  };

  mat1 = mat1.unaryExpr([&init_fn](const SharedValue& val) { return init_fn(val); });
  mat2 = mat2.unaryExpr([&init_fn](const SharedValue& val) { return init_fn(val); });
  
  for (auto _ : state) {
    Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic> res = mat1 * mat2;
    benchmark::DoNotOptimize(res);
  }
}

class FloatWrapper {
public:
    explicit FloatWrapper(float value = 0.0F) : value_(value) {}

    float get() const { return value_; }
    void set(float value) { value_ = value; }

    FloatWrapper operator+(const FloatWrapper& other) const {
        return FloatWrapper(value_ + other.value_);
    }

    FloatWrapper operator-(const FloatWrapper& other) const {
        return FloatWrapper(value_ - other.value_);
    }

    FloatWrapper operator*(const FloatWrapper& other) const {
        return FloatWrapper(value_ * other.value_);
    }

    FloatWrapper operator/(const FloatWrapper& other) const {
        return FloatWrapper(value_ / other.value_);
    }

    FloatWrapper& operator+=(const FloatWrapper& other) {
        value_ += other.value_;
        return *this;
    }

    FloatWrapper& operator-=(const FloatWrapper& other) {
        value_ -= other.value_;
        return *this;
    }

    FloatWrapper& operator*=(const FloatWrapper& other) {
        value_ *= other.value_;
        return *this;
    }

    FloatWrapper& operator/=(const FloatWrapper& other) {
        value_ /= other.value_;
        return *this;
    }

    bool operator==(const FloatWrapper& other) const {
        return value_ == other.value_;
    }

    bool operator!=(const FloatWrapper& other) const {
        return value_ != other.value_;
    }

    bool operator<(const FloatWrapper& other) const {
        return value_ < other.value_;
    }

    bool operator<=(const FloatWrapper& other) const {
        return value_ <= other.value_;
    }

    bool operator>(const FloatWrapper& other) const {
        return value_ > other.value_;
    }

    bool operator>=(const FloatWrapper& other) const {
        return value_ >= other.value_;
    }

private:
    float value_;
};

void BM_MatrixMultiplicationFloat(benchmark::State& state) {
    Eigen::Matrix<FloatWrapper, Eigen::Dynamic, Eigen::Dynamic> mat1(kRows1, kCols1); 
    Eigen::Matrix<FloatWrapper, Eigen::Dynamic, Eigen::Dynamic> mat2(kCols1, kCols2);

    auto init_fn = [](float) {
        return FloatWrapper((2.0F * (static_cast<float>(rand()) / RAND_MAX) - 1.0F) / 1000);
    };

    mat1 = mat1.unaryExpr([&init_fn](FloatWrapper) { return init_fn(0); });
    mat2 = mat2.unaryExpr([&init_fn](FloatWrapper) { return init_fn(0); });
  
    for (auto _ : state) {
        Eigen::Matrix<FloatWrapper, Eigen::Dynamic, Eigen::Dynamic> res = mat1 * mat2;
        benchmark::DoNotOptimize(res);
    }
}

BENCHMARK(BM_MatrixMultiplicationSharedValue);
BENCHMARK(BM_MatrixMultiplicationFloat);

}

BENCHMARK_MAIN();
