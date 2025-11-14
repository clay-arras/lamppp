project(tests LANGUAGES CXX)

# FETCH CONTENT ---------------------------------
include(FetchContent)
set(FETCHCONTENT_TRY_FIND_PACKAGE TRUE)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.12.0
  GIT_SHALLOW    TRUE
  FIND_PACKAGE_ARGS CONFIG
)
FetchContent_MakeAvailable(pybind11)

# EXECUTABLES ---------------------------------

pybind11_add_module(pylamp core.cpp)
target_link_libraries(pylamp PRIVATE lamppp)

if(LMP_ENABLE_COVERAGE)
  target_compile_options(pylamp PRIVATE --coverage -O0 -g)
  target_link_options(pylamp PRIVATE --coverage)
endif()

set_target_properties(pylamp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG   "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}"
)
