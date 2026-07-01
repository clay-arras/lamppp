project(inductor_lib LANGUAGES CXX)

set(INDUCTOR_SOURCES)

if (LMP_ENABLE_CUDA)
  enable_language(CUDA)
  list(APPEND INDUCTOR_SOURCES
    nvrtc/nvrtc_backend.cpp
    nvrtc/fused_graph.cpp
    nvrtc/codegen.cpp
  )
endif()

if (NOT INDUCTOR_SOURCES)
  set(_inductor_empty ${CMAKE_CURRENT_BINARY_DIR}/inductor_empty.cpp)
  file(WRITE ${_inductor_empty} "")
  list(APPEND INDUCTOR_SOURCES ${_inductor_empty})
endif()

add_library(inductor_core SHARED ${INDUCTOR_SOURCES})

target_link_libraries(inductor_core PUBLIC lamp3_compile_config tensor_core)

# Apply coverage flags if enabled
if(LMP_ENABLE_COVERAGE)
  target_compile_options(inductor_core PRIVATE --coverage -O0 -g)
  target_link_options(inductor_core PRIVATE --coverage)
endif()

if (LMP_ENABLE_CUDA)
  set_target_properties(inductor_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  target_link_libraries(inductor_core
    PUBLIC CUDA::cudart CUDA::nvrtc CUDA::cuda_driver
  )
endif()

target_compile_definitions(inductor_core PUBLIC)

target_include_directories(inductor_core
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

set(PYTHON_EXECUTABLE ${PYTHON_EXECUTABLE} PARENT_SCOPE)
