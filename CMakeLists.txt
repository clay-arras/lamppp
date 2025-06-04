cmake_minimum_required(VERSION 3.24)
project(lamppp LANGUAGES CXX)

option(ENABLE_CUDA "Compile with CUDA support" OFF)
option(ENABLE_COVERAGE "Enable code coverage" OFF)

if(ENABLE_CUDA)
  add_definitions(-DENABLE_CUDA)
endif()

# # Coverage configuration
# if(ENABLE_COVERAGE)
#   if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
#     # Coverage flags
#     set(COVERAGE_COMPILE_FLAGS "--coverage -O0 -g")
#     set(COVERAGE_LINK_FLAGS "--coverage")
    
#     # Apply to CMAKE_CXX_FLAGS
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COVERAGE_COMPILE_FLAGS}")
#     set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${COVERAGE_LINK_FLAGS}")
#     set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${COVERAGE_LINK_FLAGS}")
    
#     message(STATUS "Code coverage enabled")
#   else()
#     message(WARNING "Code coverage requested but compiler is not GNU GCC or Clang")
#   endif()
# endif()

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

# Enabled for shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# CORE TARGETS ---------------------------------
add_subdirectory(src/tensor)
add_subdirectory(src/autograd)
add_library(lamppp INTERFACE)
target_link_libraries(lamppp INTERFACE autograd_core tensor_core)

target_compile_options(lamppp
  INTERFACE
    $<$<CONFIG:Release>:-O3 -march=native -ffast-math -DNDEBUG>
    $<$<CONFIG:Debug>:-O3 -march=native -ffast-math -DLMP_DEBUG -fopenmp -g>
)

# # Override optimization for coverage builds
# if(ENABLE_COVERAGE)
#   target_compile_options(lamppp
#     INTERFACE
#       -O0 -g --coverage
#   )
# endif()

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
  set(DOXYGEN_INPUT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include")
  set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/docs")
  set(PROJECT_NAME ${CMAKE_PROJECT_NAME})
  
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in 
    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile  
    @ONLY
  )

  doxygen_add_docs(
    docs
    "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/include"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DOXYFILE ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/docs
  )
endif()

# # COVERAGE TARGETS ---------------------------------
# if(ENABLE_COVERAGE)
#   # Find gcovr for coverage reports
#   find_program(GCOVR_PATH gcovr)
#   if(GCOVR_PATH)
#     # Coverage cleanup target
#     add_custom_target(coverage_clean
#       COMMAND find ${CMAKE_BINARY_DIR} -name "*.gcda" -delete
#       COMMAND find ${CMAKE_BINARY_DIR} -name "*.gcno" -delete
#       COMMENT "Cleaning coverage data"
#     )
    
#     # Coverage report target (HTML)
#     add_custom_target(coverage_html
#       COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/coverage
#       COMMAND ${GCOVR_PATH}
#         --root=${CMAKE_SOURCE_DIR}
#         --object-directory=${CMAKE_BINARY_DIR}
#         --exclude-unreachable-branches
#         --exclude-throw-branches
#         --exclude='.*_deps.*'
#         --exclude='.*external.*'
#         --exclude='.*build.*cuda.*'
#         --gcov-ignore-errors=no_working_dir_found
#         --html --html-details
#         --output=${CMAKE_BINARY_DIR}/coverage/index.html
#       WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
#       COMMENT "Generating HTML coverage report"
#     )
    
#     # Coverage report target (XML for CI)
#     add_custom_target(coverage_xml
#       COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/coverage
#       COMMAND ${GCOVR_PATH}
#         --root=${CMAKE_SOURCE_DIR}
#         --object-directory=${CMAKE_BINARY_DIR}
#         --exclude-unreachable-branches
#         --exclude-throw-branches
#         --exclude='.*_deps.*'
#         --exclude='.*external.*'
#         --exclude='.*build.*cuda.*'
#         --gcov-ignore-errors=no_working_dir_found
#         --xml
#         --output=${CMAKE_BINARY_DIR}/coverage/coverage.xml
#       WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
#       COMMENT "Generating XML coverage report"
#     )
    
#     # Combined coverage target that runs tests and generates report
#     add_custom_target(coverage
#       COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target coverage_clean
#       COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target autograd_tests
#       COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target tensor_tests
#       COMMAND ${CMAKE_BINARY_DIR}/tests/autograd_tests --gtest_filter="*/0"
#       COMMAND ${CMAKE_BINARY_DIR}/tests/tensor_tests --gtest_filter="*/0"
#       COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target coverage_html
#       COMMENT "Running tests and generating coverage report"
#     )
    
#     message(STATUS "Coverage targets added: coverage, coverage_html, coverage_xml, coverage_clean")
#   else()
#     message(WARNING "gcovr not found. Coverage reports will not be available.")
#   endif()
# endif()