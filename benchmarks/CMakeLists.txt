project(benchmarks LANGUAGES CXX)

if(LMP_ENABLE_CUDA)
  enable_language(CUDA)
endif()

# FETCH CONTENT ---------------------------------
include(FetchContent)
set(FETCHCONTENT_TRY_FIND_PACKAGE TRUE)

FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        v1.9.0
  GIT_SHALLOW    TRUE
  FIND_PACKAGE_ARGS CONFIG
)
FetchContent_MakeAvailable(benchmark)

# EXECUTABLES ---------------------------------
add_executable(reg_bench_long lamp3/reg_bench_long.cpp)
target_link_libraries(reg_bench_long
  PRIVATE lamp3 benchmark::benchmark
)

add_executable(reg_bench_short lamp3/reg_bench_short.cpp)
target_link_libraries(reg_bench_short
  PRIVATE lamp3 benchmark::benchmark
)