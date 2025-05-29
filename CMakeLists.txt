cmake_minimum_required(VERSION 3.24)
project(lamppp LANGUAGES CXX CUDA)

# CORE CONFIG ---------------------------------
set(PYTHON_VENV_PATH "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")

if(EXISTS ${PYTHON_VENV_PATH})
    set(Python3_EXECUTABLE ${PYTHON_VENV_PATH} CACHE FILEPATH "Path to Python3 executable from Poetry venv" FORCE)
    message(STATUS "Attempting to use Python executable from Poetry venv: ${Python3_EXECUTABLE}")
else()
    message(WARNING "Poetry virtual environment Python not found at ${PYTHON_VENV_PATH}. CMake will attempt to find system Python. This may lead to version mismatches.")
endif()

# Expected python version is >=3.11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "CMake is using Python executable: ${Python3_EXECUTABLE}")
message(STATUS "Python3 version: ${Python3_VERSION_STRING}")
set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})

# Setting C++ and CUDA versions
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enabled for shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)              


# FETCH CONTENT ---------------------------------
include(FetchContent)
set(FETCHCONTENT_TRY_FIND_PACKAGE TRUE)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
  GIT_SHALLOW    TRUE
  FIND_PACKAGE_ARGS CONFIG 
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        v1.9.0
  GIT_SHALLOW    TRUE
  FIND_PACKAGE_ARGS CONFIG
)
FetchContent_MakeAvailable(benchmark)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.12.0
  GIT_SHALLOW    TRUE
  FIND_PACKAGE_ARGS CONFIG
)
FetchContent_MakeAvailable(pybind11)

# CORE TARGETS ---------------------------------
add_subdirectory(src/tensor)
add_subdirectory(src/autograd)
add_library(lamppp INTERFACE)
target_link_libraries(lamppp INTERFACE autograd_core tensor_core)

target_compile_options(lamppp
  INTERFACE
    $<$<CONFIG:Release>:-O3 -march=native -ffast-math -DNDEBUG>
    $<$<CONFIG:Debug>:-O3 -march=native -ffast-math -DLMP_DEBUG -fopenmp>
)

# EXECUTABLES ---------------------------------
enable_testing()

add_executable(playground tests/playground.cpp)
target_link_libraries(playground PRIVATE lamppp)

add_executable(autograd_tests tests/autograd_tests.cpp)
target_link_libraries(autograd_tests
  PRIVATE autograd_core GTest::gtest_main GTest::gmock
)
include(GoogleTest)
gtest_discover_tests(autograd_tests)

add_executable(tensor_tests tests/tensor_tests.cpp)
target_link_libraries(tensor_tests
  PRIVATE tensor_core GTest::gtest_main GTest::gmock
)
include(GoogleTest)
gtest_discover_tests(tensor_tests)

add_executable(bench_ops benchmarks/bench_ops.cpp)
target_link_libraries(bench_ops
  PRIVATE lamppp benchmark::benchmark
)

pybind11_add_module(lamppp_module tests/autograd_pybind.cpp)
target_link_libraries(lamppp_module PRIVATE lamppp)

set_target_properties(lamppp_module PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG   "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}"
)

# STATIC ANALYSIS ---
find_program(CLANG_FORMAT_EXE NAMES clang-format)
find_program(CLANG_TIDY_EXE NAMES clang-tidy)

if(CLANG_TIDY_EXE)
  set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
endif()

# FORMATTING --
if(CLANG_FORMAT_EXE)
  file(GLOB_RECURSE ALL_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/tensor/*.[ch]pp" "${CMAKE_CURRENT_SOURCE_DIR}/src/tensor/*.[ch]u"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/autograd/*.[ch]pp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.[ch]pp" 
    "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*.[ch]pp" 
  )

  add_custom_target(format
    COMMAND ${CLANG_FORMAT_EXE}
            -i
            ${ALL_SOURCE}
    COMMENT "Apply clang-format to all sources"
  )
endif()
