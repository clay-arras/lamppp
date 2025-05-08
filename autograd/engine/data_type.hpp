#pragma once

#include <cstdint>
#ifndef DATA_TYPE_H
#define DATA_TYPE_H

enum class DataType : uint8_t { Int32 = 0, Float32 = 1, Float64 = 2 };

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

#endif  // DATA_TYPE_H