#pragma once

#ifndef _VARIABLE_OPS_H_
#define _VARIABLE_OPS_H_

#include "variable.hpp"

namespace autograd {
inline namespace ops {

Variable operator+(const Variable& a, const Variable& b);
Variable operator-(const Variable& a, const Variable& b);
Variable operator*(const Variable& a, const Variable& b);
Variable operator/(const Variable& a, const Variable& b);

Variable operator==(const Variable& a, const Variable& b);
Variable operator!=(const Variable& a, const Variable& b);
Variable operator>=(const Variable& a, const Variable& b);
Variable operator<=(const Variable& a, const Variable& b);
Variable operator>(const Variable& a, const Variable& b);
Variable operator<(const Variable& a, const Variable& b);

Variable matmul(const Variable& a, const Variable& b);
Variable transpose(const Variable& a);
Variable sum(const Variable& a, int axis);
Variable max(const Variable& a, int axis);

Variable exp(const Variable& a);
Variable log(const Variable& a);
Variable relu(const Variable& a);

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