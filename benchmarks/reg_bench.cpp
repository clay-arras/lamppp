#include <benchmark/benchmark.h>
#include <vector>
#include <memory>
#include "op_defs.hpp"

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);

    std::vector<lmp::tensor::DataType> dtypes = {lmp::tensor::DataType::Float32};
    std::vector<lmp::tensor::DeviceType> devices = {
        lmp::tensor::DeviceType::CPU, 
        lmp::tensor::DeviceType::CUDA
    };
    std::vector<std::vector<size_t>> shapes = {
        {128, 128},
        {256, 256}, 
        {512, 512},
        {1024, 1024}
    };

    // Create operators
    std::vector<std::unique_ptr<OperatorBase>> operators;
    operators.push_back(std::make_unique<AddOp>());

    // Register all combinations
    for (const auto& op : operators) {
        for (const auto& dtype : dtypes) {
            for (const auto& device : devices) {
                for (const auto& shape : shapes) {
                    OperatorConfig config{.shape = shape, .device = device, .dtype = dtype};
                    op->register_benchmarks(config);
                }
            }
        }
    }

    benchmark::RunSpecifiedBenchmarks();
    return 0;
} 