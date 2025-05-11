#pragma once

#include <cstdint>
#include <ostream>

enum class DeviceType : uint8_t { CPU = 0, CUDA = 1 };

inline std::ostream& operator<<(std::ostream& os, DeviceType device) {
  switch (device) {
    case DeviceType::CPU:
      os << "CPU";
      break;
    case DeviceType::CUDA:
      os << "CUDA";
      break;
    default:
      os << "Unknown DeviceType";
      break;
  }
  return os;
}
