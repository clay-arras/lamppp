project(benchmarks LANGUAGES CXX)

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
add_executable(bench_ops bench_ops.cpp)
target_link_libraries(bench_ops
  PRIVATE lamppp benchmark::benchmark
)