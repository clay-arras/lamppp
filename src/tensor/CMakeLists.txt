project(tensor_lib LANGUAGES CXX CUDA)

set(TENSOR_SOURCES
  functions/expand_ops.cpp
  functions/matrix_ops.cpp
  functions/unary_ops.cpp
  functions/reduct_ops.cpp
  native/resize.cu
  native/empty.cu
  native/fill.cu
  native/copy.cu
  cuda/matrix.cu
  cuda/offset_util.cu
  cuda/unary.cu
  cuda/reduct.cu
  cuda/expand.cu
  cuda/kernels.cu
  cpu/meta_handler.cpp
  cpu/offset_util.cpp
  cpu/unary.cpp
  cpu/kernels.cpp
  fill_like.cpp
  tensor.cpp
  tensor_impl.cpp
  storage.cpp
  align_utils.cpp
)

add_library(tensor_core SHARED ${TENSOR_SOURCES})

set_target_properties(tensor_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON 
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_definitions(tensor_core PUBLIC)

target_link_libraries(tensor_core
  PUBLIC CUDA::cudart
)

target_include_directories(
  tensor_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set(PYTHON_EXECUTABLE ${PYTHON_EXECUTABLE} PARENT_SCOPE)