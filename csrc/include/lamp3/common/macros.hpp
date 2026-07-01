/*
Copied from this blog post: https://sillycross.github.io/2022/04/01/2022-04-01/
I have no idea how this works, but I need it for some explicit instatiations and I don't want to 
include the whole Boost library just for the boost/preprocessor stuff. All credit goes to sillycross
*/

// google linter hates this code :p
// NOLINTBEGIN

/// @internal
// Macro utility 'PP_FOR_EACH' and 'PP_FOR_EACH_CARTESIAN_PRODUCT'
// Apply macro to all elements in a list, or all elements in the Cartesian product of multiple lists
// Requires C++20.
//
// ----------------
// PP_FOR_EACH(m, ...)
//     Expands to m(l) for all l in list
//     m(l) will expand further if 'm' is a macro
//
// Example: PP_FOR_EACH(m, 1, 2, 3) expands to m(1) m(2) m(3)
//
// ----------------
// PP_FOR_EACH_CARTESIAN_PRODUCT(m, lists...)
//
//     Expands to m(l) for all l in List1 * ... * ListN where * denotes Cartesian product.
//     m(l) will expand further if 'm' is a macro
//     The terms are enumerated in lexical order of the lists
//
// Example:
//     PP_FOR_EACH_CARTESIAN_PRODUCT(m, (1,2), (A,B), (x,y))
// expands to
//     m(1,A,x) m(1,A,y) m(1,B,x) m(1,B,y) m(2,A,x) m(2,A,y) m(2,B,x) m(2,B,y)
//
// The implementation is inspired by the following articles:
//     https://www.scs.stanford.edu/~dm/blog/va-opt.html
//     https://github.com/pfultz2/Cloak/wiki/C-Preprocessor-tricks,-tips,-and-idioms
//     https://stackoverflow.com/questions/2308243/macro-returning-the-number-of-arguments-it-is-given-in-c
//

#define LMP_CAT(a, ...) LMP_PRIMITIVE_CAT(a, __VA_ARGS__)
#define LMP_PRIMITIVE_CAT(a, ...) a##__VA_ARGS__

// LMP_IS_EXACTLY_TWO_ARGS(...)
// Expands to 1 if exactly two parameters is passed in, otherwise expands to 0
//
#define LMP_IS_EXACTLY_TWO_ARGS(...) \
  LMP_GET_FIRST_ARG(__VA_OPT__(LMP_IS_TWO_ARGS_IMPL1(__VA_ARGS__), ) 0)
#define LMP_GET_FIRST_ARG(a, ...) a
#define LMP_IS_TWO_ARGS_IMPL1(p1, ...) \
  LMP_GET_FIRST_ARG(__VA_OPT__(LMP_IS_TWO_ARGS_IMPL2(__VA_ARGS__), ) 0)
#define LMP_IS_TWO_ARGS_IMPL2(p1, ...) LMP_GET_FIRST_ARG(__VA_OPT__(0, ) 1)

// LMP_IS_ZERO(x): Expands to 1 if x is 0, otherwise expands to 0
//
#define LMP_IS_ZERO(x) \
  LMP_IS_EXACTLY_TWO_ARGS(LMP_PRIMITIVE_CAT(LMP_IS_ZERO_IMPL_, x))
#define LMP_IS_ZERO_IMPL_0 0, 0

// LMP_EXPAND_LIST((list)): Expands to list (i.e. the parenthesis is removed)
//
#define LMP_EXPAND_LIST_IMPL(...) __VA_ARGS__
#define LMP_EXPAND_LIST(...) LMP_EXPAND_LIST_IMPL __VA_ARGS__
#define LMP_ADD_COMMA_IF_NONEMPTY(...) __VA_OPT__(__VA_ARGS__, )
#define LMP_EXPAND_LIST_TRAIL_COMMA(...) \
  LMP_ADD_COMMA_IF_NONEMPTY(LMP_EXPAND_LIST_IMPL __VA_ARGS__)

// LMP_IF_EQUAL_ZERO(cond)((true_br), (false_br))
// Expands to true_br if cond is 0, otherwise expands to false_br
//
#define LMP_IF_EQUAL_ZERO(cond) \
  LMP_CAT(LMP_IF_EQUAL_ZERO_IMPL_, LMP_IS_ZERO(cond))
#define LMP_IF_EQUAL_ZERO_IMPL_1(truebr, falsebr) LMP_EXPAND_LIST(truebr)
#define LMP_IF_EQUAL_ZERO_IMPL_0(truebr, falsebr) LMP_EXPAND_LIST(falsebr)

// LMP_INC(x) increments x
//
#define LMP_INC(x) LMP_PRIMITIVE_CAT(LMP_INC_, x)
#define LMP_INC_0 1
#define LMP_INC_1 2
#define LMP_INC_2 3
#define LMP_INC_3 4
#define LMP_INC_4 5
#define LMP_INC_5 6
#define LMP_INC_6 7
#define LMP_INC_7 8
#define LMP_INC_8 9
#define LMP_INC_9 10
#define LMP_INC_10 11
#define LMP_INC_11 12
#define LMP_INC_12 13
#define LMP_INC_13 14
#define LMP_INC_14 15
#define LMP_INC_15 16
#define LMP_INC_16 17
#define LMP_INC_17 18
#define LMP_INC_18 19
#define LMP_INC_19 19

// LMP_DEC(x) decrements x
//
#define LMP_DEC(x) LMP_PRIMITIVE_CAT(LMP_DEC_, x)
#define LMP_DEC_0 0
#define LMP_DEC_1 0
#define LMP_DEC_2 1
#define LMP_DEC_3 2
#define LMP_DEC_4 3
#define LMP_DEC_5 4
#define LMP_DEC_6 5
#define LMP_DEC_7 6
#define LMP_DEC_8 7
#define LMP_DEC_9 8
#define LMP_DEC_10 9
#define LMP_DEC_11 10
#define LMP_DEC_12 11
#define LMP_DEC_13 12
#define LMP_DEC_14 13
#define LMP_DEC_15 14
#define LMP_DEC_16 15
#define LMP_DEC_17 16
#define LMP_DEC_18 17
#define LMP_DEC_19 18

// LMP_COUNT_ARGS(...): returns the total number of arguments
// https://stackoverflow.com/questions/2308243/macro-returning-the-number-of-arguments-it-is-given-in-c
//
#define LMP_COUNT_ARGS(...) \
  LMP_COUNT_ARGS_IMPL(__VA_ARGS__ __VA_OPT__(, ) LMP_COUNT_ARGS_IMPL_SEQ())
#define LMP_COUNT_ARGS_IMPL(...) LMP_COUNT_ARGS_IMPL_GET_64TH_ARG(__VA_ARGS__)
#define LMP_COUNT_ARGS_IMPL_GET_64TH_ARG(                                      \
    a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16,     \
    a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, \
    a32, a33, a34, a35, a36, a37, a38, a39, a40, a41, a42, a43, a44, a45, a46, \
    a47, a48, a49, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a60, a61, \
    a62, a63, N, ...)                                                          \
  N
#define LMP_COUNT_ARGS_IMPL_SEQ()                                             \
  63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, \
      44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, \
      26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9,  \
      8, 7, 6, 5, 4, 3, 2, 1, 0

// Takes "a list where the first element is also a list" and a value, perform a rotation like below:
// (L1, v2, v3, v4), v5 => (v2, v3, v4, v5), expanded(L1)
//
#define LMP_LIST_ROTATION(a, b) LMP_LIST_ROTATION_IMPL1(LMP_EXPAND_LIST(a), b)
#define LMP_LIST_ROTATION_IMPL1(...) LMP_LIST_ROTATION_IMPL2(__VA_ARGS__)
#define LMP_LIST_ROTATION_IMPL2(a, ...) (__VA_ARGS__), LMP_EXPAND_LIST(a)

#define LMP_PARENS ()

#define LMP_CARTESIAN_IMPL_ENTRY(dimLeft, macro, vpack, ...)              \
  __VA_OPT__(LMP_CARTESIAN_IMPL_AGAIN_2 LMP_PARENS(dimLeft, macro, vpack, \
                                                   __VA_ARGS__))

#define LMP_CARTESIAN_EMIT_ONE(macro, ...) macro(__VA_ARGS__)
#define LMP_CARTESIAN_EMIT_ONE_PARAMS(vpack, vfirst) \
  LMP_EXPAND_LIST_TRAIL_COMMA(vpack) vfirst

#define LMP_CARTESIAN_IMPL(dimLeft, macro, vpack, vfirst, ...)              \
  LMP_IF_EQUAL_ZERO(dimLeft)                                                \
  ((LMP_CARTESIAN_EMIT_ONE(macro,                                           \
                           LMP_CARTESIAN_EMIT_ONE_PARAMS(vpack, vfirst))),  \
   (LMP_CARTESIAN_IMPL_ENTRY_AGAIN_2 LMP_PARENS(                            \
       LMP_DEC(dimLeft), macro, LMP_LIST_ROTATION(vpack, vfirst))))         \
      __VA_OPT__(LMP_CARTESIAN_IMPL_AGAIN LMP_PARENS(dimLeft, macro, vpack, \
                                                     __VA_ARGS__))

#define LMP_CARTESIAN_IMPL_AGAIN_2() LMP_CARTESIAN_IMPL_AGAIN LMP_PARENS
#define LMP_CARTESIAN_IMPL_AGAIN() LMP_CARTESIAN_IMPL
#define LMP_CARTESIAN_IMPL_ENTRY_AGAIN_2() \
  LMP_CARTESIAN_IMPL_ENTRY_AGAIN LMP_PARENS
#define LMP_CARTESIAN_IMPL_ENTRY_AGAIN() LMP_CARTESIAN_IMPL_ENTRY

#define LMP_EXPAND(...) \
  LMP_EXPAND4(LMP_EXPAND4(LMP_EXPAND4(LMP_EXPAND4(__VA_ARGS__))))
#define LMP_EXPAND4(...) \
  LMP_EXPAND3(LMP_EXPAND3(LMP_EXPAND3(LMP_EXPAND3(__VA_ARGS__))))
#define LMP_EXPAND3(...) \
  LMP_EXPAND2(LMP_EXPAND2(LMP_EXPAND2(LMP_EXPAND2(__VA_ARGS__))))
#define LMP_EXPAND2(...) \
  LMP_EXPAND1(LMP_EXPAND1(LMP_EXPAND1(LMP_EXPAND1(__VA_ARGS__))))
#define LMP_EXPAND1(...) __VA_ARGS__

// FOR_EACH implementation from https://www.scs.stanford.edu/~dm/blog/va-opt.html
// See comment at beginning of this file
//
#define LMP_FOR_EACH(macro, ...) \
  __VA_OPT__(LMP_EXPAND(LMP_FOR_EACH_HELPER(macro, __VA_ARGS__)))
#define LMP_FOR_EACH_HELPER(macro, a1, ...) \
  macro(a1) __VA_OPT__(LMP_FOR_EACH_AGAIN LMP_PARENS(macro, __VA_ARGS__))
#define LMP_FOR_EACH_AGAIN() LMP_FOR_EACH_HELPER
#define LMP_FOR_EACH_INDIRECTION(...) LMP_FOR_EACH(__VA_ARGS__)

// FOR_EACH_CARTESIAN_PRODUCT(macro, lists...)
// See comment at beginning of this file
//
#define LMP_FOR_EACH_CARTESIAN_PRODUCT(macro, list1, ...)                 \
  LMP_EXPAND(LMP_CARTESIAN_IMPL_ENTRY(LMP_COUNT_ARGS(__VA_ARGS__), macro, \
                                      (__VA_ARGS__), LMP_EXPAND_LIST(list1)))

/// @endinternal

// NOLINTEND