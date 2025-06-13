project(tests LANGUAGES CXX)

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
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.12.0
  GIT_SHALLOW    TRUE
  FIND_PACKAGE_ARGS CONFIG
)
FetchContent_MakeAvailable(pybind11)

# EXECUTABLES ---------------------------------
enable_testing()

add_executable(autograd_tests unit/autograd_tests.cpp)
target_link_libraries(autograd_tests
  PRIVATE autograd_core GTest::gtest_main GTest::gmock
)

add_executable(tensor_tests unit/tensor_tests.cpp)
target_link_libraries(tensor_tests
  PRIVATE tensor_core GTest::gtest_main GTest::gmock
)

pybind11_add_module(lamppp_module stress/autograd_pybind.cpp)
target_link_libraries(lamppp_module PRIVATE lamppp)

if(ENABLE_COVERAGE)
  target_compile_options(tensor_tests PRIVATE --coverage -O0 -g)
  target_link_options(tensor_tests PRIVATE --coverage)
  target_compile_options(autograd_tests PRIVATE --coverage -O0 -g)
  target_link_options(autograd_tests PRIVATE --coverage)
  target_compile_options(lamppp_module PRIVATE --coverage -O0 -g)
  target_link_options(lamppp_module PRIVATE --coverage)
endif()

include(GoogleTest)
gtest_discover_tests(tensor_tests PROPERTIES LABELS "library_tests")
gtest_discover_tests(autograd_tests PROPERTIES LABELS "library_tests")

set_target_properties(lamppp_module PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG   "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}"
)
