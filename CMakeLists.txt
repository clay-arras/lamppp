cmake_minimum_required(VERSION 3.10)
project(autograd_cpp)

set(CMAKE_CXX_CLANG_TIDY "clang-tidy")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(benchmark REQUIRED)

add_library(autograd_core
    autograd/engine/engine.cpp
    autograd/engine/grad.cpp
    autograd/engine/wrapper_engine.cpp
    autograd/engine/value_pool.cpp
    autograd/nn/nn.cpp
    autograd/nn/fast_layer.cpp
    autograd/util/csv_reader.cpp
    test/cpp/dummy_value.cpp
    test/cpp/dummy_mem_pool.cpp
)

target_include_directories(autograd_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIRS}
)

add_executable(test_engine test/cpp/test_engine.cpp)
target_link_libraries(test_engine autograd_core)

add_executable(test_nn test/cpp/test_nn.cpp)
target_link_libraries(test_nn autograd_core)

add_executable(mnist autograd/mnist/mnist.cpp)
target_link_libraries(mnist autograd_core)

add_executable(fast_mnist autograd/mnist/fast_mnist.cpp)
target_link_libraries(fast_mnist autograd_core)

add_executable(test_mnist test/cpp/test_mnist.cpp)
target_link_libraries(test_mnist autograd_core)

add_executable(test_fast_mnist test/cpp/test_fast_mnist.cpp)
target_link_libraries(test_fast_mnist autograd_core)

add_executable(test_mem_pool test/cpp/test_mem_pool.cpp)
target_link_libraries(test_mem_pool autograd_core)

add_executable(bench_nn test/benchmarks/bench_nn.cpp)
target_link_libraries(bench_nn autograd_core benchmark::benchmark)

add_executable(bench_fast_layer test/benchmarks/bench_fast_layer.cpp)
target_link_libraries(bench_fast_layer autograd_core benchmark::benchmark)

add_executable(bench_mat test/benchmarks/bench_mat.cpp)
target_link_libraries(bench_mat autograd_core benchmark::benchmark)

find_program(CLANG_FORMAT_EXE NAMES clang-format)
if (NOT CLANG_FORMAT_EXE)
  message(FATAL_ERROR "clang-format not found!")
endif()

file(GLOB_RECURSE ALL_SOURCE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
)

add_custom_target(format
    COMMAND ${CLANG_FORMAT_EXE} -i ${ALL_SOURCE_FILES}
    COMMENT "Running clang-format on source files..."
)
