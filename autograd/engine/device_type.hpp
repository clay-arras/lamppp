#pragma once

#include <ostream>
#ifndef DEVICE_TYPE_H
#define DEVICE_TYPE_H

#include <cstdint>

enum class DeviceType : uint8_t { CPU = 0, CUDA = 1 };

std::ostream& operator<<(std::ostream& os, DeviceType device) {
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

#endif  // DEVICE_TYPE_H
