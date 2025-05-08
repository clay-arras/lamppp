#include <iostream>
#include <typeinfo>

template <typename INNER, typename OUTER>
void kernel() {
  std::cout << "kernel<INNER=" << typeid(INNER).name()
            << ", OUTER=" << typeid(OUTER).name() << "> invoked." << std::endl;
}

int main() {
  // To test the macros, we'll call some of the kernel functions
  // that the macros declare. The specific functions available will
  // depend on the content of your "test.def" file.

  // Example calls, assuming test.def contains X(int), X(float), and X(double):
  kernel<int, int>();
  kernel<float, int>();
  kernel<double, int>();

  kernel<int, float>();
  kernel<float, float>();
  kernel<double, float>();

  kernel<int, double>();
  kernel<float, double>();
  kernel<double, double>();

  std::cout << "Test of macro-generated kernel calls completed." << std::endl;
  return 0;
}

template <typename T, typename U, typename V>
void test_Kernel_def() {
  std::cout << "Type T: " << typeid(T).name()
            << ", Type U: " << typeid(U).name()
            << ", Type V: " << typeid(V).name() << std::endl;
}

// clang-format off
// #define U_TEMPLATE(TYPES, OUTER) void kernel<TYPES, OUTER>();

// #define V_TEMPLATE(TYPES)                             \
//   _Pragma("push_macro(\"X\")")                        \
//   #undef X                                            \
//   #define X(THIS)                                     \
//       U_TEMPLATE(THIS, TYPES)                         \
//   #include "test.def"                                 \
//   _Pragma("pop_macro(\"X\")")

// #define X V_TEMPLATE
// #include "test.def"
// #undef X

// #undef U_TEMPLATE
// #undef V_TEMPLATE


// #define DECLARE_FOR_U(U)                 \
//     #define X(V) template void kernel<U,V>(); \
//     #include "test.def"                  \
//     #undef  X

// /* outer definition walks U */
// #define X DECLARE_FOR_U
// #include "test.def"      // → 9 explicit instantiations
// #undef X
// #undef DECLARE_FOR_U

// clang-format on
