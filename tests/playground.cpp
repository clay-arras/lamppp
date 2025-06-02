#include <cassert>
#include <iostream>
#include <sstream>

namespace detail {

class AssertStream {
 public:
  AssertStream(const char* file, int line, const char* expr) {
    os_ << file << ':' << line << ": ASSERT(" << expr << ") failed: ";
  }

  template <class T>
  AssertStream& operator<<(T&& v) {
    os_ << std::forward<T>(v);
    return *this;
  }

  [[noreturn]] void trigger() const { throw std::runtime_error(os_.str()); }

 private:
  std::ostringstream os_;
};

struct Voidify {
  template <class T>
  void operator&(T&& stream) const {
    stream.trigger();
  }
};

}  // namespace detail

#define ASSERT(cond)             \
  (cond) ? (void)0               \
         : ::detail::Voidify() & \
               ::detail::AssertStream(__FILE__, __LINE__, #cond)

int main() {
  ASSERT(1 == 1) << "hello!!!";
  ASSERT(1 == 2) << "hello";
}
