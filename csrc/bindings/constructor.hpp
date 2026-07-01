#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lamp3/tensor/core.hpp"
#include "lamp3/autograd/core.hpp"
#include "lamp3/common/macros.hpp"
#include "lamp3/tensor/data_type.hpp"

namespace py = pybind11;

inline void init_constructor(py::module_& m) {
    m.def("zeros", &lmp::autograd::zeros);
    m.def("ones", &lmp::autograd::ones);
    m.def("rand", &lmp::autograd::rand);
    m.def("randn", &lmp::autograd::randn);
    // TODO(root): need to figure out some workaround to bind tensor. because it's templated, maybe use std::any and type erase??
}