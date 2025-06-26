#pragma once

#include <cstdint>
#include <ostream>

namespace lmp::tensor {

/**
 * @note: this is just a useful utility, since it's the highest priority type.
 * i.e. if you have Operation(AnyType, Float64), it'll typecast to Float64
 * @see DataType
 */
using Scalar = double;

/**
* @brief simple dataType enum
*
* @note this class works deeply in conjunction with LMP_DISPATCH_ALL_TYPES
* to enable type-agnostic tensors
*
* @see type_upcast for more details on how type upcasting works
*/
enum class DataType : uint8_t {
  Bool = 0,
  Int16 = 1,
  Int32 = 2,
  Int64 = 3,
  Float32 = 4,
  Float64 = 5
};

/// @internal
/**
* @brief simple template to convert from a concrete type (like int) to the enum type, 
* Int32. example usage: TypeMeta<T>::value, where T is a template
*/
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

/// @endinternal

/**
* @brief simple type-upcasting system which leverages the values in enums
* 
* @details this library does not, and will not, support complex numbers; therefore, 
* we have this simple system of hardcoding priority where Bool = 0 = lowest priority, and
* Float64 = 5 = highest priority. To decide which type to upcast to, we just take the maximum.
*
*/
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

#define LMP_X_TYPES(_) \
  _(bool)            \
  _(int16_t)         \
  _(int)             \
  _(int64_t)         \
  _(float)           \
  _(double)

#define LMP_LIST_TYPES \
  (bool, int16_t, int, int64_t, float, double)
  