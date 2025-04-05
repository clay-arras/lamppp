#ifndef _FLOAT_EIGEN_SUPPORT_H_
#define _FLOAT_EIGEN_SUPPORT_H_

#include <Eigen/Core>
#include <cmath>
#include "test/cpp/dummy_value.h"

namespace Eigen {
template <>
struct NumTraits<Float> : NumTraits<float> {
  using Real = Float;
  using NonInteger = Float;
  using Nested = Float;

  enum {
    IsComplex = 0,
    IsInteger = 0,
    IsSigned = 1,
    RequireInitialization = 1,
    ReadCost = 1,
    AddCost = 1,
    MulCost = 1
  };

  static Real epsilon() { return Float(1e-10F); }
  static Real dummy_precision() { return Float(1e-10F); }
  static Real highest() { return Float(std::numeric_limits<float>::max()); }
  static Real lowest() { return Float(std::numeric_limits<float>::lowest()); }
};
}  // namespace Eigen

#endif