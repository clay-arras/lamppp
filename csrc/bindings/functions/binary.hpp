#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lamppp/autograd/core.hpp"

namespace py = pybind11;

using lmp::tensor::DataType;

void init_binary(py::module_& m) {
    m.def("eq", &lmp::autograd::ops::eq);
    m.def("ne", &lmp::autograd::ops::ne);
    m.def("ge", &lmp::autograd::ops::ge);
    m.def("le", &lmp::autograd::ops::le);
    m.def("gt", &lmp::autograd::ops::gt);
    m.def("lt", &lmp::autograd::ops::lt);
}

