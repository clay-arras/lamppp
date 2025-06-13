#include <benchmark/benchmark.h>
#include "bench_binary.cuh"

static void BM_CudaVectorAdd(benchmark::State& state) {
    CudaDataPtr d_a = create_cuda_data_ptr(VECTOR_SIZE);
    CudaDataPtr d_b = create_cuda_data_ptr(VECTOR_SIZE);
    CudaDataPtr d_result = create_cuda_data_ptr(VECTOR_SIZE);
    
    launch_initialize_random_array(d_a.data_as_float(), VECTOR_SIZE, 1234);
    launch_initialize_random_array(d_b.data_as_float(), VECTOR_SIZE, 5678);
    
    for (auto _ : state) {
        launch_vector_add(
            d_a.data_as_float(), 
            d_b.data_as_float(), 
            d_result.data_as_float(), 
            VECTOR_SIZE
        );
        benchmark::DoNotOptimize(d_result.data());
    }
}

BENCHMARK(BM_CudaVectorAdd);

BENCHMARK_MAIN();
