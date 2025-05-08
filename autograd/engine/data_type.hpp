#pragma once

#include <cstdint>
#include <ostream>

enum class DataType : uint8_t { Int32 = 0, Float32 = 1, Float64 = 2 };

template <class T>
struct TypeMeta;

template <>
struct TypeMeta<int> {
  static constexpr DataType value = DataType::Int32;
};
template <>
struct TypeMeta<float> {
  static constexpr DataType value = DataType::Float32;
};
template <>
struct TypeMeta<double> {
  static constexpr DataType value = DataType::Float64;
};

inline std::ostream& operator<<(std::ostream& os, DataType dtype) {
  switch (dtype) {
    case DataType::Int32:
      os << "Int32";
      break;
    case DataType::Float32:
      os << "Float32";
      break;
    case DataType::Float64:
      os << "Float64";
      break;
    default:
      os << "Unknown DataType";
      break;
  }
  return os;
}

/*
IMPORTANT NOTE: if expand to other data types in the future, do something like: 
enum DTypeRank {
  Bool    = 0,
  Int16   = 1,
  Int32   = 2,
  Int64   = 3,
  Float16 = 4,
  Float32 = 5,
  Float64 = 6
};

where dtype promotion just gets the larger of the types
*/
