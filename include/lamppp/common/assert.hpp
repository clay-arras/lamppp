#pragma once

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <sstream>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace lmp {
namespace detail {

template<typename Derived>
class BaseStream {
public:
    template<class T>
    Derived& operator<<(T&& v) {
        os_ << std::forward<T>(v);
        return static_cast<Derived&>(*this);
    }

protected:
    std::ostringstream os_;
};

class CheckStream : public BaseStream<CheckStream> {
public:
    CheckStream(const char* file, int line, const char* func, const char* expr) {
        os_ << "Lamppp: Runtime error thrown at " << file << ':' << line 
            << " in " << func << ". CHECK(" << expr << ") failed: ";
    }

    [[noreturn]] void trigger() const {
        std::cerr << os_.str() << std::endl;
        throw std::runtime_error(os_.str());
    }
};

#ifdef LMP_DEBUG

class AssertStream : public BaseStream<AssertStream> {
public:
    AssertStream(const char* file, int line, const char* func, const char* expr) {
        os_ << "Lamppp: Internal assertion failure at " << file << ':' << line 
            << " in " << func << ". ASSERT(" << expr << ") failed: ";
    }

    [[noreturn]] void trigger() const {
        std::cerr << os_.str() << std::endl;
        assert(false);
    }
};

#ifdef ENABLE_CUDA
class CudaAssertStream : public BaseStream<CudaAssertStream> {
public:
    CudaAssertStream(const char* file, int line, const char* func, cudaError_t err) {
        os_ << "Lamppp: CUDA error at " << file << ':' << line 
            << " in " << func << ". Error: " << cudaGetErrorString(err) << ". ";
    }

    [[noreturn]] void trigger() const {
        std::cerr << os_.str() << std::endl;
        assert(false);
    }
};
#endif

#endif

struct Voidify {
    template<class T>
    void operator&(T&& stream) const {
        stream.trigger();    
    }
};

} // namespace detail
} // namespace lmp

// ignoring some error with warning: control reaches end of non-void function [-Wreturn-type]
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type"

#define LMP_CHECK(cond) \
    (cond) ? (void)0 : ::lmp::detail::Voidify() & ::lmp::detail::CheckStream(__FILE__, __LINE__, __func__, #cond)

#pragma GCC diagnostic pop

#ifdef LMP_DEBUG

#define LMP_INTERNAL_ASSERT(cond) \
    (cond) ? (void)0 : ::lmp::detail::Voidify() & ::lmp::detail::AssertStream(__FILE__, __LINE__, __func__, #cond)

#ifdef ENABLE_CUDA
#define LMP_CUDA_ASSERT(call)                                                   \
  [&]() {                                                                       \
    cudaError_t _err = (call);                                                  \
    return (_err == cudaSuccess)                                                \
           ? (void)0                                                            \
           : ::lmp::detail::Voidify() &                                         \
             ::lmp::detail::CudaAssertStream(__FILE__, __LINE__, __func__, _err); \
  }()
#endif


#define LMP_PRINT(fmt, ...)                                              \
  fprintf(stderr, "[%s:%d] %s: " fmt "\n", __FILE__, __LINE__, __func__, \
          ##__VA_ARGS__)

#else

namespace lmp::detail {
struct NullStream {
    template<typename T> NullStream& operator<<(T&&) { return *this; }
    void trigger() const {}
};
} 
#define LMP_INTERNAL_ASSERT(cond) \
    true ? (void)0 : ::lmp::detail::Voidify() & ::lmp::detail::NullStream()
#define LMP_CUDA_ASSERT(call) \
    (call) ? (void)0 : ::lmp::detail::Voidify() & ::lmp::detail::NullStream()

#endif