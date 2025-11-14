#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lamppp/autograd/core.hpp"

namespace py = pybind11;

using lmp::tensor::DataType;

void init_matrix(py::module_& m) {
    m.def("matmul", &lmp::autograd::ops::matmul);
    m.def("transpose", &lmp::autograd::ops::transpose);
}

