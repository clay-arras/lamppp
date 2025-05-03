#ifndef DATA_TYPE_DISPATCH_H
#define DATA_TYPE_DISPATCH_H

#include <cassert>

#define DISPATCH_ALL_TYPES(TYPE, ...)  \
  [&]{                                        \
    switch(TYPE) {                            \
      case DataType::Float32: {                \
        using scalar_t = float;                \
        return __VA_ARGS__();                  \
      }                                        \
      case DataType::Float64: {                \
        using scalar_t = double;               \
        return __VA_ARGS__();                  \
      }                                        \
      case DataType::Int32: {                  \
        using scalar_t = int;                  \
        return __VA_ARGS__();                  \
      }                                        \
      default:                               \
        assert(false); \
    }                                        \
  }()

#endif // DATA_TYPE_DISPATCH_H
