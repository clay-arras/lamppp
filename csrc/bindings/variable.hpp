#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lamppp/tensor/core.hpp"
#include "lamppp/autograd/core.hpp"
#include "lamppp/common/macros.hpp"
#include "lamppp/tensor/data_type.hpp"

namespace py = pybind11;

using lmp::autograd::Variable;
using lmp::tensor::DataType;
using lmp::tensor::DeviceType;
using lmp::tensor::Tensor;

#define LMP_VAR_OVERLOAD(x) .def("__" #x "__", &lmp::autograd::ops::x)
#define LMP_VAR_FUNCTION(x) .def(#x, &lmp::autograd::ops::x)

void init_variable_overloads(py::class_<Variable> &cls) {
  cls
    LMP_FOR_EACH_CARTESIAN_PRODUCT(LMP_VAR_OVERLOAD, 
      (add, sub, mul, pow, matmul, neg, abs, eq, lt, le, gt, ge, ne))
    LMP_FOR_EACH_CARTESIAN_PRODUCT(LMP_VAR_FUNCTION, 
      (to, expand_dims, squeeze, reshape, exp, log, sqrt, abs, sin, 
        cos, tan, clamp, sum, max, min, prod))
    .def("__truediv__", &lmp::autograd::ops::div)
    .def("T", &lmp::autograd::ops::transpose);
}

#undef LMP_VAR_OVERLOAD
#undef LMP_VAR_FUNCTION

void init_variable(py::module_& m) {
  auto cls = py::class_<Variable>(m, "_Variable")
      .def(py::init<Tensor, bool>(), py::arg("data"),
           py::arg("requires_grad"))
      .def_property("data", &Variable::data, nullptr)
      .def_property("grad", &Variable::grad, nullptr)
      .def_property("grad_fn", &Variable::grad_fn, nullptr)
      .def_property("requires_grad", &Variable::requires_grad, nullptr)
      .def("backward", &Variable::backward)
      .def("tolist", [](const Variable& self) {
        return self.data().to_vector<lmp::tensor::Scalar>();
      })
      .def("zero_grad", &Variable::zero_grad)
      .def("__repr__", [](const Variable& self) {
        std::ostringstream oss;
        oss << self;
        return oss.str();
      });
  
  init_variable_overloads(cls);
}