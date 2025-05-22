#pragma once

#include <boost/stacktrace.hpp>
#include <cassert>
#include <iostream>
#include <stdexcept>

#define LMP_CHECK(cond, message)                                         \
  do {                                                                   \
    if (!(cond)) {                                                       \
      std::cerr << "Lamppp: Runtime error thrown at " << __FILE__ << ":" \
                << __LINE__ << " in " << __func__                        \
                << ". Stacktrace: " << boost::stacktrace::stacktrace()   \
                << std::endl;                                            \
      throw std::runtime_error((message));                               \
    }                                                                    \
  } while (0)

#ifdef LMP_DEBUG

#define LMP_INTERNAL_ASSERT(cond, message)                                     \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "Lamppp: Internal assertion failure at " << __FILE__ << ":" \
                << __LINE__ << " in " << __func__                              \
                << ". Stacktrace: " << boost::stacktrace::stacktrace()         \
                << std::endl;                                                  \
      assert((cond) && (message));                                             \
    }                                                                          \
  } while (0)

#define LMP_CUDA_ASSERT(call, message)                                     \
  do {                                                                     \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
      std::cerr << "Lamppp: CUDA error at " << __FILE__ << ":" << __LINE__ \
                << " in " << __func__                                      \
                << ". Error: " << cudaGetErrorString(err)                  \
                << ". Stacktrace: " << boost::stacktrace::stacktrace()     \
                << std::endl;                                              \
      assert(false && (message));                                          \
    }                                                                      \
  } while (0)

#define LMP_PRINT(fmt, ...)                                              \
  fprintf(stderr, "[%s:%d] %s: " fmt "\n", __FILE__, __LINE__, __func__, \
          ##__VA_ARGS__)

#else

// in case LMP_CUDA_ASSERT contains running code
#define LMP_INTERNAL_ASSERT(cond, ...)
#define LMP_CUDA_ASSERT(cond, ...) (cond)

#endif