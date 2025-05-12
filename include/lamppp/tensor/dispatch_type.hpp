#pragma once

#include <cassert>

#define LMP_DISPATCH_ALL_TYPES(TYPE, ...)  \
  [&] {                                    \
    switch (TYPE) {                        \
      case DataType::Bool: {               \
        using scalar_t = bool;             \
        return __VA_ARGS__();              \
      }                                    \
      case DataType::Int16: {              \
        using scalar_t = int16_t;          \
        return __VA_ARGS__();              \
      }                                    \
      case DataType::Int32: {              \
        using scalar_t = int;              \
        return __VA_ARGS__();              \
      }                                    \
      case DataType::Int64: {              \
        using scalar_t = int64_t;          \
        return __VA_ARGS__();              \
      }                                    \
      case DataType::Float32: {            \
        using scalar_t = float;            \
        return __VA_ARGS__();              \
      }                                    \
      case DataType::Float64: {            \
        using scalar_t = double;           \
        return __VA_ARGS__();              \
      }                                    \
      default:                             \
        assert(false && "Type not found"); \
    }                                      \
  }()
