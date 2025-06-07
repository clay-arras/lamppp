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
add_executable(reg_bench lamppp/reg_bench.cpp)
target_link_libraries(reg_bench
  PRIVATE lamppp benchmark::benchmark
)