project(nets_lib LANGUAGES CXX)

set(nets_SOURCES
  module.cpp
  any.cpp
  layers/linear.cpp
  layers/activation.cpp
  layers/dropout.cpp
  layers/container.cpp
)

add_library(nets_core SHARED ${nets_SOURCES})

if(LMP_ENABLE_COVERAGE)
  target_compile_options(nets_core PRIVATE --coverage -O0 -g)
  target_link_options(nets_core PRIVATE --coverage)
endif()

target_link_libraries(nets_core
  PUBLIC tensor_core autograd_core lamppp_compile_config
)

target_include_directories(nets_core
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)