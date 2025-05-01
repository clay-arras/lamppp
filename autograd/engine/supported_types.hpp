#pragma once

#include "autograd/engine/scalar.hpp"
#ifndef _SUPPORTED_TYPES_H_
#define _SUPPORTED_TYPES_H_

#include <variant>

#define X(TYPE) TYPE,
using any_type = std::variant<
#include "autograd/engine/supported_types.def"
    autograd::Scalar>;
#undef X

#endif  // _SUPPORTED_TYPES_H_
