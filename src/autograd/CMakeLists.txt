project(autograd_lib LANGUAGES CXX)

set(AUTOGRAD_SOURCES
  constructor.cpp
  variable.cpp
  functions/binary_ops.cpp
  functions/matrix_ops.cpp
  functions/unary_ops.cpp
  functions/reduct_ops.cpp
  functions/basic_ops.cpp
)

add_library(autograd_core ${AUTOGRAD_SOURCES})

target_link_libraries(autograd_core
  PUBLIC tensor_core 
)

target_include_directories(autograd_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include> # For installing headers
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
