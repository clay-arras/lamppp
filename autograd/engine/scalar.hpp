#pragma once

#ifndef _SCALAR_H_
#define _SCALAR_H_

namespace autograd {

using scalar_internal_type = long double;
class Scalar {
 public:
#define X(TYPE) \
  Scalar(TYPE data) : _data(static_cast<scalar_internal_type>(data)){};
#include "autograd/engine/supported_types.def"
#undef X

#define X(TYPE) \
  operator TYPE() const { return static_cast<TYPE>(_data); };
#include "autograd/engine/supported_types.def"
#undef X

 private:
  scalar_internal_type _data;
};

}  // namespace autograd

#endif  // _SCALAR_H_
