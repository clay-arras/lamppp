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

add_executable(playground playground.cpp)
target_link_libraries(playground PRIVATE lamppp)

add_executable(autograd_tests autograd_tests.cpp)
target_link_libraries(autograd_tests
  PRIVATE autograd_core GTest::gtest_main GTest::gmock
)

# Apply coverage flags if enabled
if(ENABLE_COVERAGE)
  target_compile_options(autograd_tests PRIVATE --coverage -O0 -g)
  target_link_options(autograd_tests PRIVATE --coverage)
endif()

include(GoogleTest)
gtest_discover_tests(autograd_tests)

add_executable(tensor_tests tensor_tests.cpp)
target_link_libraries(tensor_tests
  PRIVATE tensor_core GTest::gtest_main GTest::gmock
)

# Apply coverage flags if enabled
if(ENABLE_COVERAGE)
  target_compile_options(tensor_tests PRIVATE --coverage -O0 -g)
  target_link_options(tensor_tests PRIVATE --coverage)
endif()

include(GoogleTest)
gtest_discover_tests(tensor_tests)

pybind11_add_module(lamppp_module autograd_pybind.cpp)
target_link_libraries(lamppp_module PRIVATE lamppp)

set_target_properties(lamppp_module PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG   "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}"
)
