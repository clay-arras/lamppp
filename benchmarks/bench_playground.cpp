#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>
#include "lamppp/lamppp.hpp"
#include "lamppp/tensor/cuda/kernels.cuh"
#include "lamppp/tensor/cpu/meta_handler.hpp"
#include "lamppp/tensor/cuda/unary.cuh"
#include "lamppp/tensor/dispatch_type.hpp"
#include "lamppp/tensor/tensor.hpp"
#include "lamppp/tensor/tensor_impl.hpp"


/*
--------------------------------------------------------------------
Benchmark                          Time             CPU   Iterations
--------------------------------------------------------------------
BM_Unary_Kernel               189759 ns       189755 ns         3291
BM_Dispatched_CUDA            193539 ns       193536 ns         4004
BM_Device_Dispatch_CUDA       186926 ns       186926 ns         4034
BM_TensorImpl_Accessor          57.0 ns         57.0 ns     11659651
BM_TensorImpl_Accessor_Op    1439320 ns       539407 ns          962
BM_Tensor_Op                 1495861 ns       516389 ns         1000


BM_Unary_Kernel               172710 ns       172708 ns         3605
BM_Dispatched_CUDA            173623 ns       173602 ns         3845
BM_Device_Dispatch_CUDA       176609 ns       176621 ns         4320
BM_TensorImpl_Accessor          34.1 ns         34.1 ns     20290761
BM_TensorImpl_Accessor_Op     848224 ns       366310 ns         2095
BM_Tensor_Op                  840628 ns       359002 ns         1921
*/

// static void BM_Playground(benchmark::State& state) {
//     std::vector<float> data(512 * 512, 1.12F); 
//     for (auto _ : state) {
//         // lmp::tensor::TensorImpl cpy(orig);
//         // benchmark::DoNotOptimize(cpy);
//         lmp::tensor::TensorImpl orig(data, {512, 512}, lmp::tensor::DeviceType::CUDA, lmp::tensor::DataType::Float32);
//         benchmark::DoNotOptimize(orig);
//     }
// }

static void BM_Unary_Kernel(benchmark::State& state) {
    std::vector<float> data(512 * 512, 1.12F); 
    lmp::tensor::TensorImpl orig(data, {512, 512}, lmp::tensor::DeviceType::CUDA, lmp::tensor::DataType::Float32);
    lmp::tensor::detail::UnaryMetaHandler meta(&orig);
    for (auto _ : state) {
        lmp::tensor::detail::cuda::unary_dispatch_handler<lmp::tensor::detail::cuda::SinFunctor> (meta);
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(meta.out());
    }
}

static void BM_Dispatched_CUDA(benchmark::State& state) {
    std::vector<float> data(512 * 512, 1.12F); 
    lmp::tensor::TensorImpl orig(data, {512, 512}, lmp::tensor::DeviceType::CUDA, lmp::tensor::DataType::Float32);
    for (auto _ : state) {
        lmp::tensor::TensorImpl res = lmp::tensor::detail::cuda::sin_cuda(orig);
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(res);
    }
}

static void BM_Device_Dispatch_CUDA(benchmark::State& state) {
    std::vector<float> data(512 * 512, 1.12F); 
    lmp::tensor::TensorImpl orig(data, {512, 512}, lmp::tensor::DeviceType::CUDA, lmp::tensor::DataType::Float32);
    for (auto _ : state) {
        lmp::tensor::TensorImpl res = lmp::sin_stub()(orig.device(), orig);
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(res);
    }
}

static void BM_TensorImpl_Accessor_Op(benchmark::State& state) {
    std::vector<float> data(512 * 512, 1.12F); 
    lmp::tensor::TensorImpl orig(data, {512, 512}, lmp::tensor::DeviceType::CUDA, lmp::tensor::DataType::Float32);
    lmp::Tensor ten(data, {512, 512}, lmp::tensor::DeviceType::CUDA, lmp::tensor::DataType::Float32);
    for (auto _ : state) {
        lmp::tensor::TensorImpl res = lmp::sin_stub()(orig.device(), *lmp::tensor::detail::UnsafeTensorAccessor::getImpl(ten));
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(res);
    }
}

static void BM_TensorImpl_Accessor(benchmark::State& state) {
    std::vector<float> data(512 * 512, 1.12F); 
    lmp::tensor::TensorImpl orig(data, {512, 512}, lmp::tensor::DeviceType::CUDA, lmp::tensor::DataType::Float32);
    lmp::Tensor ten(data, {512, 512}, lmp::tensor::DeviceType::CUDA, lmp::tensor::DataType::Float32);
    for (auto _ : state) {
        auto res = *lmp::tensor::detail::UnsafeTensorAccessor::getImpl(ten);
        benchmark::DoNotOptimize(res);
    }
}

static void BM_Tensor_Op(benchmark::State& state) {
    std::vector<float> data(512 * 512, 1.12F); 
    lmp::Tensor ten(data, {512, 512}, lmp::tensor::DeviceType::CUDA, lmp::tensor::DataType::Float32);
    for (auto _ : state) {
        lmp::Tensor out = lmp::sin(ten);
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(out);
    }
}

static void BM_Stack_vs_Heap_TensorImpl(benchmark::State& state) {
    std::vector<float> data(512 * 512, 1.12F); 
    
    // Create TensorImpl on heap manually (like Tensor does)
    auto heap_impl = std::make_shared<lmp::tensor::TensorImpl>(data, std::vector<size_t>{512, 512}, 
                        lmp::tensor::DeviceType::CUDA, lmp::tensor::DataType::Float32);
    
    for (auto _ : state) {
        lmp::tensor::TensorImpl res = lmp::sin_stub()(heap_impl->device(), *heap_impl);
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(res);
    }
}

static void BM_Pointer_Indirection_Test(benchmark::State& state) {
    std::vector<float> data(512 * 512, 1.12F); 
    lmp::tensor::TensorImpl orig(data, {512, 512}, lmp::tensor::DeviceType::CUDA, lmp::tensor::DataType::Float32);
    lmp::tensor::TensorImpl* ptr = &orig;  // Stack object via pointer
    
    for (auto _ : state) {
        lmp::tensor::TensorImpl res = lmp::sin_stub()(ptr->device(), *ptr);
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(res);
    }
}

static void BM_Member_Access_Pattern(benchmark::State& state) {
    std::vector<float> data(512 * 512, 1.12F); 
    lmp::Tensor ten(data, {512, 512}, lmp::tensor::DeviceType::CUDA, lmp::tensor::DataType::Float32);
    auto impl = lmp::tensor::detail::UnsafeTensorAccessor::getImpl(ten);
    
    for (auto _ : state) {
        // Access the same members that TensorMetaHandler accesses
        auto device = impl->device();
        auto shape = impl->shape();
        auto type = impl->type();
        auto numel = impl->numel();
        auto strides = impl->strides();
        benchmark::DoNotOptimize(device);
        benchmark::DoNotOptimize(shape);
        benchmark::DoNotOptimize(type);
        benchmark::DoNotOptimize(numel);
        benchmark::DoNotOptimize(strides);
    }
}

static void BM_GPU_Pointer_Same_Check(benchmark::State& state) {
    std::vector<float> data(512 * 512, 1.12F); 
    lmp::tensor::TensorImpl orig(data, {512, 512}, lmp::tensor::DeviceType::CUDA, lmp::tensor::DataType::Float32);
    lmp::Tensor ten(data, {512, 512}, lmp::tensor::DeviceType::CUDA, lmp::tensor::DataType::Float32);
    auto impl = lmp::tensor::detail::UnsafeTensorAccessor::getImpl(ten);
    
    // Check if GPU pointers are the same
    printf("Stack GPU ptr: %p\n", orig.data());
    printf("Heap GPU ptr: %p\n", impl->data());
    printf("Are they same? %s\n", (orig.data() == impl->data()) ? "YES" : "NO");
    
    state.SkipWithError("Pointer comparison - check output");
}

static void BM_Fresh_vs_Reused_Objects(benchmark::State& state) {
    std::vector<float> data(512 * 512, 1.12F); 
    
    for (auto _ : state) {
        // Create new Tensor every iteration (like BM_TensorImpl_Accessor_Op does)
        lmp::Tensor ten(data, {512, 512}, lmp::tensor::DeviceType::CUDA, lmp::tensor::DataType::Float32);
        lmp::tensor::TensorImpl res = lmp::sin_stub()(ten.device(), 
                                     *lmp::tensor::detail::UnsafeTensorAccessor::getImpl(ten));
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(res);
    }
}

static void BM_GPU_Memory_Investigation(benchmark::State& state) {
    std::vector<float> data(512 * 512, 1.12F); 
    
    // Pre-warm GPU memory allocator
    for (int i = 0; i < 10; i++) {
        lmp::Tensor dummy(data, {512, 512}, lmp::tensor::DeviceType::CUDA, lmp::tensor::DataType::Float32);
    }
    
    lmp::tensor::TensorImpl orig(data, {512, 512}, lmp::tensor::DeviceType::CUDA, lmp::tensor::DataType::Float32);
    
    for (auto _ : state) {
        lmp::tensor::TensorImpl res = lmp::sin_stub()(orig.device(), orig);
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(res);
    }
}

static void BM_Memory_Locality_Test(benchmark::State& state) {
    std::vector<float> data(512 * 512, 1.12F); 
    
    for (auto _ : state) {
        // Create Tensor, but don't reuse the variable
        lmp::Tensor ten(data, {512, 512}, lmp::tensor::DeviceType::CUDA, lmp::tensor::DataType::Float32);
        // Force a different memory allocation pattern
        lmp::tensor::TensorImpl res = lmp::sin_stub()(ten.device(), 
                                     *lmp::tensor::detail::UnsafeTensorAccessor::getImpl(ten));
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(res);
        // Tensor 'ten' gets destroyed here, freeing GPU memory
    }
}

static void BM_TensorMetaHandler_Creation(benchmark::State& state) {
    std::vector<float> data(512 * 512, 1.12F); 
    lmp::Tensor ten(data, {512, 512}, lmp::tensor::DeviceType::CUDA, lmp::tensor::DataType::Float32);
    auto impl = lmp::tensor::detail::UnsafeTensorAccessor::getImpl(ten);
    
    for (auto _ : state) {
        lmp::tensor::detail::UnaryMetaHandler meta(impl.get());
        benchmark::DoNotOptimize(meta);
    }
}

// Test if it's the Storage allocation specifically
static void BM_Storage_Allocation_Pattern(benchmark::State& state) {
    std::vector<float> data(512 * 512, 1.12F); 
    lmp::Tensor ten(data, {512, 512}, lmp::tensor::DeviceType::CUDA, lmp::tensor::DataType::Float32);
    
    for (auto _ : state) {
        // This mimics what TensorMetaHandler does
        lmp::tensor::Storage out_st(512 * 512 * sizeof(float), lmp::tensor::DeviceType::CUDA);
        benchmark::DoNotOptimize(out_st);
    }
}

// // Test immediate deallocation theory
// static void BM_Force_Immediate_Dealloc_Scope(benchmark::State& state) {
//     std::vector<float> data(512 * 512, 1.12F); 
    
//     for (auto _ : state) {
//         lmp::tensor::TensorImpl res;
//         {
//             // Force destruction immediately after use
//             lmp::Tensor ten(data, {512, 512}, lmp::tensor::DeviceType::CUDA, lmp::tensor::DataType::Float32);
//             res = lmp::sin_stub()(ten.device(), *lmp::tensor::detail::UnsafeTensorAccessor::getImpl(ten));
//             // ten gets destroyed here, shared_ptr refcount goes to 0
//         }
//         cudaDeviceSynchronize();
//         benchmark::DoNotOptimize(res);
//     }
// }

static void BM_Force_Immediate_Dealloc_Reset(benchmark::State& state) {
    std::vector<float> data(512 * 512, 1.12F); 
    
    for (auto _ : state) {
        auto ten_ptr = std::make_shared<lmp::tensor::TensorImpl>(data, std::vector<size_t>{512, 512}, 
                                                                lmp::tensor::DeviceType::CUDA, 
                                                                lmp::tensor::DataType::Float32);
        lmp::tensor::TensorImpl res = lmp::sin_stub()(ten_ptr->device(), *ten_ptr);
        ten_ptr.reset(); // Force immediate deallocation
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(res);
    }
}

static void BM_Manual_Memory_Management(benchmark::State& state) {
    std::vector<float> data(512 * 512, 1.12F); 
    
    for (auto _ : state) {
        // Create and immediately use, then manually destroy
        auto* ten_raw = new lmp::tensor::TensorImpl(data, {512, 512}, 
                                                   lmp::tensor::DeviceType::CUDA, 
                                                   lmp::tensor::DataType::Float32);
        lmp::tensor::TensorImpl res = lmp::sin_stub()(ten_raw->device(), *ten_raw);
        delete ten_raw; // Immediate cleanup
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(res);
    }
}

static void BM_Reuse_Tensor_Object(benchmark::State& state) {
    std::vector<float> data(512 * 512, 1.12F); 
    // Create once, reuse many times
    lmp::Tensor ten(data, {512, 512}, lmp::tensor::DeviceType::CUDA, lmp::tensor::DataType::Float32);
    
    for (auto _ : state) {
        lmp::tensor::TensorImpl res = lmp::sin_stub()(ten.device(), 
                                     *lmp::tensor::detail::UnsafeTensorAccessor::getImpl(ten));
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(res);
    }
}

BENCHMARK(BM_Unary_Kernel);
BENCHMARK(BM_Dispatched_CUDA);
BENCHMARK(BM_Device_Dispatch_CUDA);
BENCHMARK(BM_TensorImpl_Accessor);
BENCHMARK(BM_TensorImpl_Accessor_Op);
BENCHMARK(BM_Tensor_Op);
BENCHMARK(BM_Stack_vs_Heap_TensorImpl);
BENCHMARK(BM_Pointer_Indirection_Test);
BENCHMARK(BM_Member_Access_Pattern);
BENCHMARK(BM_GPU_Pointer_Same_Check);
BENCHMARK(BM_Fresh_vs_Reused_Objects);
BENCHMARK(BM_GPU_Memory_Investigation);
BENCHMARK(BM_Memory_Locality_Test);
BENCHMARK(BM_TensorMetaHandler_Creation);
BENCHMARK(BM_Storage_Allocation_Pattern);
// BENCHMARK(BM_Force_Immediate_Dealloc_Scope);
BENCHMARK(BM_Force_Immediate_Dealloc_Reset);
BENCHMARK(BM_Manual_Memory_Management);
BENCHMARK(BM_Reuse_Tensor_Object);

BENCHMARK_MAIN();
