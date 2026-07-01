#pragma once

#include "lamp3/tensor/device_type.hpp"

namespace lmp::tensor::lazy {

class CaptureGuard {
public:
    explicit CaptureGuard(bool capture_enabled);
    ~CaptureGuard();

private:
bool prev;
};

bool is_capture_enabled(); 

void set_capture_enabled(bool enable);

bool should_capture(DeviceType device);

}  // namespace lmp::tensor::lazy