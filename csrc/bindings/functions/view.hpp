#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lamppp/autograd/core.hpp"

namespace py = pybind11;

using lmp::tensor::DataType;

void init_view(py::module_& m) {
    m.def("reshape", &lmp::autograd::ops::reshape);
    m.def("squeeze", &lmp::autograd::ops::squeeze);
    m.def("expand_dims", &lmp::autograd::ops::expand_dims);
    m.def("to", &lmp::autograd::ops::to);
}

