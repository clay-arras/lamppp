#include <cassert>
#include <iostream>

void func_with_args(int a, const char* str) {
  printf("Function called with args: %d, %s\n", a, str);
}

void func_no_args() {
  printf("Function called with no args\n");
}

#define CALL_FUNC(...) \
  __VA_OPT__(func_with_args(__VA_ARGS__);) __VA_OPT__(, func_no_args();)
#define TEST(...) __VA_OPT__(, __VA_ARGS__)

int main() {
  // CALL_FUNC(10, "hello"); // Calls func_with_args(10, "hello");
  // CALL_FUNC();           // Calls func_no_args();

  return 0;
}
