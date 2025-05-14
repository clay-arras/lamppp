cmake_minimum_required(VERSION 3.10)
project(autograd_cpp LANGUAGES CXX CUDA)

set(Python_EXECUTABLE ".venv/bin/python")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_CLANG_TIDY "clang-tidy")
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -DNDEBUG") # bench
add_definitions(-DEIGEN_NO_DEBUG)

find_package(Eigen3 REQUIRED)
find_package(benchmark REQUIRED)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF)
set(INSTALL_GTEST OFF)
set(BUILD_GMOCK ON)
FetchContent_MakeAvailable(googletest)

add_library(
  autograd_core
  src/autograd/constructor.cpp
  src/autograd/variable.cpp
  src/autograd/functions/binary_ops.cpp
  src/autograd/functions/matrix_ops.cpp
  src/autograd/functions/unary_ops.cpp
  src/autograd/functions/reduct_ops.cpp
  src/autograd/functions/basic_ops.cpp
  src/tensor/functions/binary_ops.cpp
  src/tensor/functions/matrix_ops.cpp
  src/tensor/functions/unary_ops.cpp
  src/tensor/functions/reduct_ops.cpp
  src/tensor/functions/basic_ops.cpp
  src/tensor/tensor_helper.cpp
  src/tensor/native/resize.cu
  src/tensor/native/empty.cu
  src/tensor/native/fill.cu
  src/tensor/native/copy.cu
  src/tensor/cuda/reduct_kern.cu
  src/tensor/cuda/binary_kern.cu
  src/tensor/cuda/matrix_kern.cu
  src/tensor/cuda/basic_kern.cu
  src/tensor/cuda/unary_kern.cu
  # src/tensor/cuda/offset_util.cu
  src/tensor/tensor.cpp
  src/tensor/tensor_impl.cpp
  src/tensor/storage.cpp
  # src/tensor/align_utils.cpp
)

set_target_properties(autograd_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_definitions(autograd_core PUBLIC)
target_link_libraries(autograd_core PUBLIC Eigen3::Eigen)

target_include_directories(
  autograd_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIRS}
                       ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

enable_testing()

add_executable(playground tests/playground.cpp)
target_link_libraries(playground autograd_core)

add_executable(autograd_tests tests/autograd_tests.cpp)
target_link_libraries(autograd_tests autograd_core GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(autograd_tests)

pybind11_add_module(lamppp tests/autograd_pybind.cpp)
target_link_libraries(lamppp PRIVATE autograd_core)

find_program(CLANG_FORMAT_EXE NAMES clang-format)
if(NOT CLANG_FORMAT_EXE)
  message(FATAL_ERROR "clang-format not found!")
endif()

file(GLOB_RECURSE ALL_SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
)

add_custom_target(
  format
  COMMAND ${CLANG_FORMAT_EXE} -i ${ALL_SOURCE_FILES}
  COMMENT "Running clang-format on source files..."
)