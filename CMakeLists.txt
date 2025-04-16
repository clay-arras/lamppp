cmake_minimum_required(VERSION 3.10)
project(autograd_cpp)

set(CMAKE_CXX_CLANG_TIDY "clang-tidy")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g")

find_package(Eigen3 REQUIRED)
find_package(benchmark REQUIRED)

add_library(autograd_core # TODO: next step: split into different bench envs 
    autograd/engine/variable.cpp
    autograd/engine/variable_ops.cpp
    autograd/engine/tensor.cpp
    autograd/engine/tensor_ops.cpp
    autograd/engine/functions/basic_ops.cpp
    autograd/engine/functions/unary_ops.cpp
    autograd/engine/functions/matrix_ops.cpp
    autograd/engine/function.cpp
    autograd/util/csv_reader.cpp
)

target_include_directories(autograd_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIRS}
)

add_executable(test_playground test/cpp/test_playground.cpp)
target_link_libraries(test_playground autograd_core benchmark::benchmark)

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
