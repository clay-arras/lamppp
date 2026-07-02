#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lamp3/autograd/core.hpp"

namespace py = pybind11;

// Binds all three existing C++ signatures: (Variable, Variable),
// (Variable, Scalar), and (Scalar, Variable) via binary_op<>.
#define LMP_BIND_BINARY_OP(name)                                       \
  m.def(#name, &lmp::autograd::ops::name, py::arg("a"), py::arg("b")); \
  m.def(#name,                                                         \
        static_cast<lmp::autograd::Variable (*)(                       \
            const lmp::autograd::Variable&, lmp::tensor::Scalar)>(     \
            &lmp::autograd::binary_op<&lmp::autograd::ops::name>),     \
        py::arg("a"), py::arg("b"));                                   \
  m.def(#name,                                                         \
        static_cast<lmp::autograd::Variable (*)(                       \
            lmp::tensor::Scalar, const lmp::autograd::Variable&)>(     \
            &lmp::autograd::binary_op<&lmp::autograd::ops::name>),     \
        py::arg("a"), py::arg("b"));

inline void init_expand(py::module_& m) {
  LMP_BIND_BINARY_OP(add)
  LMP_BIND_BINARY_OP(sub)
  LMP_BIND_BINARY_OP(mul)
  LMP_BIND_BINARY_OP(div)
  LMP_BIND_BINARY_OP(pow)
}

#undef LMP_BIND_BINARY_OP
