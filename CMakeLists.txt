cmake_minimum_required(VERSION 3.10)
project(autograd_cpp)

set(Python_EXECUTABLE ".venv/bin/python")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_CLANG_TIDY "clang-tidy")
set(CMAKE_CXX_STANDARD 17)40268309
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -DNDEBUG") # bench
add_definitions(-DEIGEN_NO_DEBUG)

find_package(Eigen3 REQUIRED)
find_package(benchmark REQUIRED)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(MKL CONFIG REQUIRED)

add_library(autograd_core # TODO: next step: split into different bench envs 
    autograd/engine/variable.cpp
    autograd/engine/variable_ops.cpp
    autograd/engine/tensor.cpp
    autograd/engine/tensor_ops.cpp
    autograd/engine/function.cpp
    autograd/engine/constructor.cpp
    autograd/engine/forward_function.cpp
    autograd/engine/functions/basic_ops.cpp
    autograd/engine/functions/unary_ops.cpp
    autograd/engine/functions/matrix_ops.cpp
    autograd/engine/functions/reduct_ops.cpp
    autograd/util/csv_reader.cpp
    autograd/util/batch_sample.cpp
)

target_compile_definitions(autograd_core
  PUBLIC EIGEN_USE_MKL_ALL 
)
target_link_libraries(autograd_core PUBLIC 
    Eigen3::Eigen 
    MKL::MKL
) 

target_include_directories(autograd_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIRS}
)

add_executable(mnist autograd/examples/mnist.cpp)
target_link_libraries(mnist autograd_core)

add_executable(bench_ops test/benchmarks/bench_ops.cpp)
target_link_libraries(bench_ops autograd_core benchmark::benchmark)

pybind11_add_module(cpp_custom_bind test/python/stress/test_stress.cpp)
target_link_libraries(cpp_custom_bind PRIVATE autograd_core)

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
