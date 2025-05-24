#pragma once

#include <cstdint>
#include <ostream>

namespace lmp::tensor {

enum class DataType : uint8_t {
  Bool = 0,
  Int16 = 1,
  Int32 = 2,
  Int64 = 3,
  Float32 = 4,
  Float64 = 5
};

template <typename T>
struct TypeMeta;

template <>
struct TypeMeta<bool> {
  static constexpr DataType value = DataType::Bool;
};
template <>
struct TypeMeta<int16_t> {
  static constexpr DataType value = DataType::Int16;
};
template <>
struct TypeMeta<int> {
  static constexpr DataType value = DataType::Int32;
};
template <>
struct TypeMeta<int64_t> {
  static constexpr DataType value = DataType::Int64;
};
template <>
struct TypeMeta<float> {
  static constexpr DataType value = DataType::Float32;
};
template <>
struct TypeMeta<double> {
  static constexpr DataType value = DataType::Float64;
};

inline DataType type_upcast(DataType a_type, DataType b_type) {
  return static_cast<DataType>(
      std::max(static_cast<uint8_t>(a_type), static_cast<uint8_t>(b_type)));
}

inline std::ostream& operator<<(std::ostream& os, DataType dtype) {
  switch (dtype) {
    case DataType::Bool:
      os << "Bool";
      break;
    case DataType::Int16:
      os << "Int16";
      break;
    case DataType::Int32:
      os << "Int32";
      break;
    case DataType::Int64:
      os << "Int64";
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

}  // namespace lmp::tensor