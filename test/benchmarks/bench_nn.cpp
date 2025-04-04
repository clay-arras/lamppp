#include <benchmark/benchmark.h>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric> 

using std::vector;

namespace {
void BM_VectorSort(benchmark::State& state) {
  vector<int> vec(state.range(0));
  std::iota(vec.begin(), vec.end(), 0); 
  std::shuffle(vec.begin(), vec.end(), std::mt19937{std::random_device{}()}); // Shuffle for randomness

  for (auto _ : state) {
    std::sort(vec.begin(), vec.end());
  }
}

BENCHMARK(BM_VectorSort)->Arg(1000)->Arg(10000)->Arg(100000);

void BM_VectorSum(benchmark::State& state) {
  std::vector<int> vec(state.range(0), 1);

  for (auto _ : state) {
    int sum = 0;
    for (int val : vec) {
      sum += val;
    }
    benchmark::DoNotOptimize(sum);
  }
}

BENCHMARK(BM_VectorSum)->Arg(1000)->Arg(10000)->Arg(100000);
}

BENCHMARK_MAIN();