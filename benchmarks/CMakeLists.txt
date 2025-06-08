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
add_executable(reg_bench_long lamppp/reg_bench_long.cpp)
target_link_libraries(reg_bench_long
  PRIVATE lamppp benchmark::benchmark
)

add_executable(reg_bench_short lamppp/reg_bench_short.cpp)
target_link_libraries(reg_bench_short
  PRIVATE lamppp benchmark::benchmark
)

add_executable(bench_test bench_test.cpp)
target_link_libraries(bench_test
  PRIVATE lamppp benchmark::benchmark
)