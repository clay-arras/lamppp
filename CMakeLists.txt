cmake_minimum_required(VERSION 3.24)
project(lamppp LANGUAGES CXX)

option(ENABLE_CUDA "Compile with CUDA support" OFF)
option(ENABLE_COVERAGE "Enable code coverage" OFF)

if(ENABLE_CUDA)
  add_definitions(-DENABLE_CUDA)
endif()

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

if (ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  set(CMAKE_CUDA_ARCHITECTURES native)
  find_package(CUDAToolkit REQUIRED)              
endif()

find_package(OpenMP REQUIRED) # TODO: make this optional

# Enabled for shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CORE TARGETS ---------------------------------
add_library(lamppp_compile_flags INTERFACE)
target_compile_options(lamppp_compile_flags
  INTERFACE
    $<$<CONFIG:Release>:-march=native;-ffast-math> # also compiles with -O3 -DNDEBUG
    $<$<CONFIG:Debug>:-march=native;-ffast-math;-DLMP_DEBUG> # -g
)

add_subdirectory(src/tensor)
add_subdirectory(src/autograd)
add_library(lamppp INTERFACE)
target_link_libraries(lamppp INTERFACE autograd_core tensor_core)

add_subdirectory(tests)
add_subdirectory(benchmarks)
add_subdirectory(examples)

# STATIC ANALYSIS ---------------------------------
find_program(CLANG_TIDY_EXE NAMES clang-tidy)
if(CLANG_TIDY_EXE)
  set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
endif()

# FORMATTING ---------------------------------
file(GLOB_RECURSE ALL_SOURCE
  "${CMAKE_CURRENT_SOURCE_DIR}/src/tensor/*.[ch]pp" "${CMAKE_CURRENT_SOURCE_DIR}/src/tensor/*.[ch]u"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/autograd/*.[ch]pp"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.[ch]pp" 
  "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*.[ch]pp" 
)

find_program(CLANG_FORMAT_EXE NAMES clang-format)
if(CLANG_FORMAT_EXE)

  add_custom_target(format
    COMMAND ${CLANG_FORMAT_EXE}
            -i
            ${ALL_SOURCE}
    COMMENT "Apply clang-format to all sources"
  )
endif()

# DOCUMENTATION ---------------------------------
find_package(Doxygen)
if (DOXYGEN_FOUND)
  set(DOXYGEN_INPUT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/docs")
  set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/docs")
  set(PROJECT_NAME ${CMAKE_PROJECT_NAME})
  
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in 
    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile  
    @ONLY
  )

  doxygen_add_docs(
    docs
    "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/docs"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DOXYFILE ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/docs
  )
endif()

# Coverage configuration  
if(ENABLE_COVERAGE)
  include(CTest)
  set(CMAKE_CXX_FLAGS_DEBUG "-fprofile-arcs -ftest-coverage")
  set(CMAKE_EXE_LINKER_FLAGS_DEBUG "--coverage")
  set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "--coverage")
  message(STATUS "Code coverage enabled")
  
  add_custom_target(coverage_clean
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/coverage.info ${CMAKE_BINARY_DIR}/python_cpp_coverage.info ${CMAKE_BINARY_DIR}/cpp_filtered.info ${CMAKE_BINARY_DIR}/python_filtered.info ${CMAKE_BINARY_DIR}/combined_coverage.info
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/coverage || true
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Cleaning up previous coverage artifacts"
  )

  add_custom_target(coverage_cpp
    COMMAND ctest --test-dir ${CMAKE_BINARY_DIR} -L library_tests -T Test -T Coverage
    COMMAND lcov --directory ${CMAKE_BINARY_DIR} --capture --output-file ${CMAKE_BINARY_DIR}/coverage.info --ignore-errors mismatch,gcov,negative,source --rc geninfo_unexecuted_blocks=1
    COMMAND lcov --extract ${CMAKE_BINARY_DIR}/coverage.info '${CMAKE_SOURCE_DIR}/src/*' '${CMAKE_SOURCE_DIR}/include/*' -o ${CMAKE_BINARY_DIR}/cpp_filtered.info
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running C++ tests and generating coverage data"
    DEPENDS coverage_clean
  )

  add_custom_target(coverage_python
    COMMAND poetry run pytest tests/
    COMMAND lcov --directory ${CMAKE_BINARY_DIR} --capture --output-file ${CMAKE_BINARY_DIR}/python_cpp_coverage.info --ignore-errors mismatch,gcov,negative,source --rc geninfo_unexecuted_blocks=1
    COMMAND lcov --extract ${CMAKE_BINARY_DIR}/python_cpp_coverage.info '${CMAKE_SOURCE_DIR}/src/*' '${CMAKE_SOURCE_DIR}/include/*' -o ${CMAKE_BINARY_DIR}/python_filtered.info
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running Python tests and capturing C++ coverage data"
    DEPENDS coverage_clean
  )

  add_custom_target(coverage
    COMMAND lcov --add-tracefile ${CMAKE_BINARY_DIR}/cpp_filtered.info --add-tracefile ${CMAKE_BINARY_DIR}/python_filtered.info --output-file ${CMAKE_BINARY_DIR}/combined_coverage.info || lcov --add-tracefile ${CMAKE_BINARY_DIR}/cpp_filtered.info --output-file ${CMAKE_BINARY_DIR}/combined_coverage.info
    COMMAND genhtml ${CMAKE_BINARY_DIR}/combined_coverage.info --output-directory ${CMAKE_BINARY_DIR}/coverage
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Combining coverage reports and generating HTML output"
    DEPENDS coverage_cpp coverage_python
  )
endif()