#include "lamp3/tensor/lazy/capture_mode.hpp"
#include "lamp3/tensor/lazy/lazy_backend.hpp"

namespace lmp::tensor {

bool thread_local capture_enabled = false;

CaptureGuard::CaptureGuard(bool capture_enabled) {
  prev = is_capture_enabled();
  set_capture_enabled(capture_enabled);
}

CaptureGuard::~CaptureGuard() {
  set_capture_enabled(prev);
}

bool is_capture_enabled() {
  return capture_enabled;
}

void set_capture_enabled(bool enable) {
  capture_enabled = enable;
}

bool should_capture(DeviceType device) {
  return (backend(device) != nullptr) && capture_enabled;
}

}  // namespace lmp::tensor
