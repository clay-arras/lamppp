#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lamp3/tensor/lazy/capture_mode.hpp"

namespace py = pybind11;

inline void init_capture_mode(py::module_& m) {
    // TODO(clay-arras): should these be in root? 
    m.def("set_capture_enabled", &lmp::tensor::lazy::set_capture_enabled);
    m.def("is_capture_enabled", &lmp::tensor::lazy::is_capture_enabled);
}