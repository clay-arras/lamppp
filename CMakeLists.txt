cmake_minimum_required(VERSION 3.17)
project(lamppp LANGUAGES CXX CUDA)

# -----------------------------------------------------------------------------
# 1) Core configuration
# -----------------------------------------------------------------------------
# find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
# message(STATUS "Using Python executable: ${Python3_EXECUTABLE}")
# set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})

# Temporary PYTHON ---
set(EXPECTED_PYTHON_VERSION_MAJOR 3)
set(EXPECTED_PYTHON_VERSION_MINOR 11)
set(PYTHON_VENV_PATH "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")

if(EXISTS ${PYTHON_VENV_PATH})
    set(Python3_EXECUTABLE ${PYTHON_VENV_PATH} CACHE FILEPATH "Path to Python3 executable from Poetry venv" FORCE)
    message(STATUS "Attempting to use Python executable from Poetry venv: ${Python3_EXECUTABLE}")
else()
    message(WARNING "Poetry virtual environment Python not found at ${PYTHON_VENV_PATH}. CMake will attempt to find system Python. This may lead to version mismatches.")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "CMake is using Python executable: ${Python3_EXECUTABLE}")
message(STATUS "Python3 version: ${Python3_VERSION_STRING}")

if(NOT Python3_VERSION_MAJOR EQUAL EXPECTED_PYTHON_VERSION_MAJOR OR NOT Python3_VERSION_MINOR EQUAL EXPECTED_PYTHON_VERSION_MINOR)
    message(WARNING "CMake found Python ${Python3_VERSION_STRING} but expected version ${EXPECTED_PYTHON_VERSION_MAJOR}.${EXPECTED_PYTHON_VERSION_MINOR} based on project requirements. This might cause issues. Ensure ${PYTHON_VENV_PATH} is correct and accessible.")
endif()

set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
# End ---

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -----------------------------------------------------------------------------
# 2) Dependencies
# -----------------------------------------------------------------------------
# find_package(Eigen3 3.3 REQUIRED NO_MODULE)     
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
    # $<$<CONFIG:Release>:-O3 -march=native -ffast-math -DNDEBUG>
    $<$<CONFIG:Debug>:-O3 -march=native -ffast-math -DLMP_DEBUG>
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

add_executable(tensor_tests tests/tensor_tests.cpp)
target_link_libraries(tensor_tests
  PRIVATE autograd_core
  PRIVATE GTest::gtest_main
)
include(GoogleTest)
gtest_discover_tests(tensor_tests)

add_executable(bench_ops benchmarks/bench_ops.cpp)
target_link_libraries(bench_ops
  PRIVATE autograd_core
  PRIVATE benchmark::benchmark
)

pybind11_add_module(lamppp tests/autograd_pybind.cpp)

set_target_properties(lamppp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG   "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}"
)

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
