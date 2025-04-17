#pragma once

#ifndef _VARIABLE_OPS_H_
#define _VARIABLE_OPS_H_

#include "variable.h"

namespace autograd {
inline namespace ops {

Variable operator+(const Variable& var, float scalar);
Variable operator-(const Variable& var, float scalar);
Variable operator*(const Variable& var, float scalar);
Variable operator/(const Variable& var, float scalar);

Variable operator+(float scalar, const Variable& var);
Variable operator-(float scalar, const Variable& var);
Variable operator*(float scalar, const Variable& var);
Variable operator/(float scalar, const Variable& var);

Variable operator==(const Variable& var, float scalar);
Variable operator!=(const Variable& var, float scalar);
Variable operator>=(const Variable& var, float scalar);
Variable operator<=(const Variable& var, float scalar);
Variable operator>(const Variable& var, float scalar);
Variable operator<(const Variable& var, float scalar);

Variable operator==(float scalar, const Variable& var);
Variable operator!=(float scalar, const Variable& var);
Variable operator>=(float scalar, const Variable& var);
Variable operator<=(float scalar, const Variable& var);
Variable operator>(float scalar, const Variable& var);
Variable operator<(float scalar, const Variable& var);

}  // namespace ops
}  // namespace autograd

#endif  // _VARIABLE_OPS_H_