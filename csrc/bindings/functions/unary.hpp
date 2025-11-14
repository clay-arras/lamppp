#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lamppp/autograd/core.hpp"

namespace py = pybind11;

using lmp::tensor::DataType;

void init_unary(py::module_& m) {
    m.def("neg", &lmp::autograd::ops::neg);
    m.def("exp", &lmp::autograd::ops::exp);
    m.def("log", &lmp::autograd::ops::log);
    m.def("sqrt", &lmp::autograd::ops::sqrt);
    m.def("abs", &lmp::autograd::ops::abs);
    m.def("sin", &lmp::autograd::ops::sin);
    m.def("cos", &lmp::autograd::ops::cos);
    m.def("tan", &lmp::autograd::ops::tan);
    m.def("clamp", &lmp::autograd::ops::clamp);
}

