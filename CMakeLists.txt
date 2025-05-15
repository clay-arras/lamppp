cmake_minimum_required(VERSION 3.17)
project(lamppp LANGUAGES CXX CUDA)

# -----------------------------------------------------------------------------
# 1) Core configuration
# -----------------------------------------------------------------------------
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -----------------------------------------------------------------------------
# 2) Dependencies
# -----------------------------------------------------------------------------
find_package(Eigen3 3.3 REQUIRED NO_MODULE)     
find_package(benchmark REQUIRED)                
find_package(pybind11 CONFIG REQUIRED)          
find_package(CUDAToolkit REQUIRED)              

# -----------------------------------------------------------------------------
# 3) Static analysis (optional)
# -----------------------------------------------------------------------------
find_program(CLANG_FORMAT_EXE NAMES clang-format)
find_program(CLANG_TIDY_EXE NAMES clang-tidy)

if(CLANG_TIDY_EXE)
  set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
endif()

# -----------------------------------------------------------------------------
# 4) Fetch GoogleTest
# -----------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# -----------------------------------------------------------------------------
# 5) Core library target
# -----------------------------------------------------------------------------
add_subdirectory(src/tensor)
add_subdirectory(src/autograd)

target_compile_options(autograd_core
  PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native -ffast-math -DNDEBUG>
)

# -----------------------------------------------------------------------------
# 6) Executables and tests
# -----------------------------------------------------------------------------
enable_testing()

add_executable(playground tests/playground.cpp)
target_link_libraries(playground PRIVATE autograd_core)

add_executable(autograd_tests tests/autograd_tests.cpp)
target_link_libraries(autograd_tests
  PRIVATE autograd_core
  PRIVATE GTest::gtest_main
)
include(GoogleTest)
gtest_discover_tests(autograd_tests)

add_executable(bench_ops benchmarks/bench_ops.cpp)
target_link_libraries(bench_ops
  PRIVATE autograd_core
  PRIVATE benchmark::benchmark
)

pybind11_add_module(lamppp tests/autograd_pybind.cpp)
target_link_libraries(lamppp
  PRIVATE autograd_core
)

# -----------------------------------------------------------------------------
# 7) Formatting target
# -----------------------------------------------------------------------------
if(CLANG_FORMAT_EXE)
  file(GLOB_RECURSE ALL_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/tensor/*.[ch]pp" "${CMAKE_CURRENT_SOURCE_DIR}/src/tensor/*.[ch]u" # .cu files for CUDA
    "${CMAKE_CURRENT_SOURCE_DIR}/src/autograd/*.[ch]pp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.[ch]pp" # Assuming tests remain top-level
    "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*.[ch]pp" # Assuming benchmarks remain top-level
  )

  add_custom_target(format
    COMMAND ${CLANG_FORMAT_EXE}
            -i
            ${ALL_SOURCE}
    COMMENT "Apply clang-format to all sources"
  )
endif()
