#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lamppp/tensor/core.hpp"

namespace py = pybind11;

inline void init_device_type(py::module_& m) {
  py::enum_<lmp::tensor::DeviceType>(m, "device")
      .value("cpu", lmp::tensor::DeviceType::CPU)
      .value("cuda", lmp::tensor::DeviceType::CUDA);
}