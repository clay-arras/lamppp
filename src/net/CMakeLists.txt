project(net_lib LANGUAGES CXX)

set(NET_SOURCES
  module.cpp
)

add_library(net_core SHARED ${NET_SOURCES})

if(ENABLE_COVERAGE)
  target_compile_options(net_core PRIVATE --coverage -O0 -g)
  target_link_options(net_core PRIVATE --coverage)
endif()

target_link_libraries(net_core
  PUBLIC tensor_core autograd_core
)

target_include_directories(net_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include> 
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
