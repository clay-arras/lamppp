#pragma once

namespace lmp::tensor {

/**
 * @note: this is just a useful utility, since it's the highest priority type.
 * i.e. if you have Operation(AnyType, Float64), it'll typecast to Float64
 * @see DataType
 */
using Scalar = double;

}  // namespace lmp::tensor