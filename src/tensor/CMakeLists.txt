project(tensor_lib LANGUAGES CXX CUDA)

set(TENSOR_SOURCES
  functions/binary_ops.cpp
  functions/matrix_ops.cpp
  functions/unary_ops.cpp
  functions/reduct_ops.cpp
  functions/basic_ops.cpp
  tensor_helper.cpp
  native/resize.cu
  native/empty.cu
  native/fill.cu
  native/copy.cu
  cuda/reduct_kern.cu
  cuda/binary_kern.cu
  cuda/matrix_kern.cu
  cuda/basic_kern.cu
  cuda/offset_util.cu
  cuda/unary.cu
  cuda/kernels.cu
  cuda/utils.cu
  tensor.cpp
  tensor_impl.cpp
  storage.cpp
  align_utils.cpp
)

add_library(tensor_core ${TENSOR_SOURCES})

set_target_properties(tensor_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON 
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_definitions(tensor_core PUBLIC EIGEN_NO_DEBUG)

target_link_libraries(tensor_core
  PUBLIC Eigen3::Eigen
  PUBLIC CUDA::cudart
)

target_include_directories(
  tensor_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${EIGEN3_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set(PYTHON_EXECUTABLE ${PYTHON_EXECUTABLE} PARENT_SCOPE)