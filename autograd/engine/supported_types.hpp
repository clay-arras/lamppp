#pragma once

#ifndef _SUPPORTED_TYPES_H_
#define _SUPPORTED_TYPES_H_

#include <variant>

#define X(TYPE) TYPE,
using any_type = std::variant<
#include "autograd/engine/supported_types.def"
void* 
>;
#undef X

#endif  // _SUPPORTED_TYPES_H_
