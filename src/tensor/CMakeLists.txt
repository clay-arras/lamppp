project(tensor_lib LANGUAGES CXX)

set(TENSOR_SOURCES
  native/expand_ops.cpp
  native/matrix_ops.cpp
  native/unary_ops.cpp
  native/reduct_ops.cpp
  native/memory_ops.cpp
  native/shape_ops.cpp
  native/conv_ops.cpp
  cpu/meta_handler.cpp
  cpu/offset_util.cpp
  cpu/unary.cpp
  cpu/conv.cpp
  cpu/binary.cpp
  cpu/expand.cpp
  cpu/reduct.cpp
  cpu/matrix.cpp
  cpu/kernels.cpp
  cpu/memory.cpp
  utils/fill_like.cpp
  tensor.cpp
  tensor_impl.cpp
  storage.cpp
  utils/align_utils.cpp
)

if (LMP_ENABLE_CUDA)
  enable_language(CUDA)
  list(APPEND TENSOR_SOURCES
    cuda/matrix.cu
    cuda/offset_util.cu
    cuda/unary.cu
    cuda/binary.cu
    cuda/reduct.cu
    cuda/expand.cu
    cuda/kernels.cu
    cuda/conv.cu
    cuda/memory.cu
  )
endif()

add_library(tensor_core SHARED ${TENSOR_SOURCES})

target_link_libraries(tensor_core PUBLIC lamppp_compile_config)

# Apply coverage flags if enabled
if(LMP_ENABLE_COVERAGE)
  target_compile_options(tensor_core PRIVATE --coverage -O0 -g)
  target_link_options(tensor_core PRIVATE --coverage)
endif()

if (LMP_ENABLE_CUDA)
  set_target_properties(tensor_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON 
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  target_link_libraries(tensor_core
    PUBLIC CUDA::cudart OpenMP::OpenMP_CXX
  )
endif()

target_compile_definitions(tensor_core PUBLIC)

target_include_directories(tensor_core
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

set(PYTHON_EXECUTABLE ${PYTHON_EXECUTABLE} PARENT_SCOPE)