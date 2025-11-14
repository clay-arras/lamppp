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

pybind11_add_module(_C core.cpp)
target_link_libraries(_C PRIVATE lamppp)

set_target_properties(_C PROPERTIES
    INSTALL_RPATH "$ORIGIN/../lib"
    BUILD_WITH_INSTALL_RPATH FALSE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

if(LMP_ENABLE_COVERAGE)
  target_compile_options(_C PRIVATE --coverage -O0 -g)
  target_link_options(_C PRIVATE --coverage)
endif()

install(TARGETS _C LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}/pylamp)