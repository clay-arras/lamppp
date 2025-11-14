#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lamppp/autograd/core.hpp"

namespace py = pybind11;

using lmp::tensor::DataType;

void init_reduct(py::module_& m) {
    m.def("sum", &lmp::autograd::ops::sum);
    m.def("max", &lmp::autograd::ops::max);
    m.def("min", &lmp::autograd::ops::min);
    m.def("prod", &lmp::autograd::ops::prod);
}

