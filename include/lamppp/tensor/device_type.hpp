#pragma once

#include <cstddef>
#include <ostream>

namespace lmp::tensor {

/**
 * @brief Device type enum
 * @note This enum is used to represent the type of device that a tensor is on.
 */
enum class DeviceType : size_t { CPU = 0, CUDA = 1, Count };

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

}  // namespace lmp::tensor