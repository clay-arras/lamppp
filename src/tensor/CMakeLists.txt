project(tensor_lib LANGUAGES CXX)

set(TENSOR_SOURCES
  native/expand_ops.cpp
  native/matrix_ops.cpp
  native/unary_ops.cpp
  native/reduct_ops.cpp
  native/memory_ops.cpp
  native/shape_ops.cpp
  cpu/meta_handler.cpp
  cpu/offset_util.cpp
  cpu/unary.cpp
  cpu/expand.cpp
  cpu/reduct.cpp
  cpu/matrix.cpp
  cpu/kernels.cpp
  cpu/memory.cpp
  fill_like.cpp
  tensor.cpp
  tensor_impl.cpp
  storage.cpp
  align_utils.cpp
)

if (ENABLE_CUDA)
  enable_language(CUDA)
  list(APPEND TENSOR_SOURCES
    cuda/matrix.cu
    cuda/offset_util.cu
    cuda/unary.cu
    cuda/reduct.cu
    cuda/expand.cu
    cuda/kernels.cu
    cuda/memory.cu
  )
endif()

add_library(tensor_core SHARED ${TENSOR_SOURCES})

# Apply coverage flags if enabled
if(ENABLE_COVERAGE)
  target_compile_options(tensor_core PRIVATE --coverage -O0 -g)
  target_link_options(tensor_core PRIVATE --coverage)
endif()

if (ENABLE_CUDA)
  set_target_properties(tensor_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON 
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  target_link_libraries(tensor_core
    PUBLIC CUDA::cudart
  )
endif()

target_compile_definitions(tensor_core PUBLIC)

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