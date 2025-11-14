#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lamppp/tensor/core.hpp"
#include "lamppp/autograd/core.hpp"

namespace py = pybind11;

using lmp::autograd::Variable;
using lmp::tensor::DataType;
using lmp::tensor::DeviceType;
using lmp::tensor::Tensor;

void init_variable(py::module_& m) {
  py::class_<Variable>(m, "_Variable")
      .def(py::init<Tensor, bool>(), py::arg("data"),
           py::arg("requires_grad"))
      .def_property("data", &Variable::data, nullptr)
      .def_property("grad", &Variable::grad, nullptr)
      .def_property("grad_fn", &Variable::grad_fn, nullptr)
      .def_property("requires_grad", &Variable::requires_grad, nullptr)
      .def("backward", &Variable::backward)
      .def("zero_grad", &Variable::zero_grad)
      .def("__repr__", [](const Variable& self) {
        std::ostringstream oss;
        oss << self;
        return oss.str();
      });
}