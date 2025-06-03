#include <cassert>

#define CALL_FUNC(...) \
  __VA_OPT__(func_with_args(__VA_ARGS__);) __VA_OPT__(, func_no_args();)
#define TEST(...) __VA_OPT__(, __VA_ARGS__)

int main() {

  return 0;
}
