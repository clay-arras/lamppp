#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lamppp/autograd/core.hpp"

namespace py = pybind11;

inline void init_expand(py::module_& m) {
    m.def("add", &lmp::autograd::ops::add);
    m.def("sub", &lmp::autograd::ops::sub);
    m.def("mul", &lmp::autograd::ops::mul);
    m.def("div", &lmp::autograd::ops::div);
    m.def("pow", &lmp::autograd::ops::pow);
}