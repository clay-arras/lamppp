#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lamppp/tensor/core.hpp"

namespace py = pybind11;

using lmp::tensor::DeviceType;

void init_device_type(py::module_& m) {
  py::enum_<DeviceType>(m, "device")
      .value("cpu", DeviceType::CPU)
      .value("cuda", DeviceType::CUDA);
}