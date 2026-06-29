project(inductor_lib LANGUAGES CXX)

set(INDUCTOR_SOURCES
  backend.cpp
)

if (LMP_ENABLE_CUDA)
  enable_language(CUDA)
endif()

add_library(inductor_core SHARED ${INDUCTOR_SOURCES})

target_link_libraries(inductor_core PUBLIC lamppp_compile_config tensor_core)

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
    PUBLIC CUDA::cudart CUDA::nvrtc
  )
endif()

target_compile_definitions(inductor_core PUBLIC)

target_include_directories(inductor_core
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

set(PYTHON_EXECUTABLE ${PYTHON_EXECUTABLE} PARENT_SCOPE)
