#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

#include "include/lamppp/autograd/core.hpp"
#include "include/lamppp/tensor/core.hpp"

namespace py = pybind11;

using lmp::autograd::Variable;
using lmp::tensor::DataType;
using lmp::tensor::Tensor;
using std::ostringstream;
using std::vector;

namespace {

Variable add_cust(const Variable& a, const Variable& b) {
  Variable c = a + b;
  c.backward();
  return c;
}

Variable sub_cust(const Variable& a, const Variable& b) {
  Variable c = a - b;
  c.backward();
  return c;
}

Variable mul_cust(const Variable& a, const Variable& b) {
  Variable c = a * b;
  c.backward();
  return c;
}

Variable div_cust(const Variable& a, const Variable& b) {
  Variable c = a / b;
  c.backward();
  return c;
}

Variable exp_cust(const Variable& a) {
  Variable c = lmp::autograd::ops::exp(a);
  c.backward();
  return c;
}

Variable log_cust(const Variable& a) {
  Variable c = lmp::autograd::ops::log(a);
  c.backward();
  return c;
}

Variable sqrt_cust(const Variable& a) {
  Variable c = lmp::autograd::ops::sqrt(a);
  c.backward();
  return c;
}

Variable abs_cust(const Variable& a) {
  Variable c = lmp::autograd::ops::abs(a);
  c.backward();
  return c;
}

Variable sin_cust(const Variable& a) {
  Variable c = lmp::autograd::ops::sin(a);
  c.backward();
  return c;
}

Variable cos_cust(const Variable& a) {
  Variable c = lmp::autograd::ops::cos(a);
  c.backward();
  return c;
}

Variable tan_cust(const Variable& a) {
  Variable c = lmp::autograd::ops::tan(a);
  c.backward();
  return c;
}

Variable clamp_cust(const Variable& a, double min_val, double max_val) {
  Variable c = lmp::autograd::ops::clamp(a, min_val, max_val);
  c.backward();
  return c;
}

Variable matmul_cust(const Variable& a, const Variable& b) {
  Variable c = lmp::autograd::ops::matmul(a, b);
  c.backward();
  return c;
}

Variable transpose_cust(const Variable& a) {
  Variable c = lmp::autograd::ops::transpose(a);
  c.backward();
  return c;
}

Variable sum_cust(const Variable& a, size_t axis) {
  Variable c = lmp::autograd::ops::sum(a, axis);
  c.backward();
  return c;
}

Variable max_cust(const Variable& a, size_t axis) {
  Variable c = lmp::autograd::ops::max(a, axis);
  c.backward();
  return c;
}

Variable min_cust(const Variable& a, size_t axis) {
  Variable c = lmp::autograd::ops::min(a, axis);
  c.backward();
  return c;
}

}  // namespace

PYBIND11_MODULE(lamppp, m) {
  py::enum_<DataType>(m, "cDataType")
      .value("Float32", DataType::Float32)
      .value("Float64", DataType::Float64)
      .value("Int32", DataType::Int32);

  py::class_<Tensor>(m, "cTensor")
      .def(py::init<const std::vector<double>, const std::vector<size_t>>(),
           py::arg("data"), py::arg("shape"))
      .def_property(
          "data",
          [](Tensor& t) -> vector<double> {
            std::span<double> sp = t.view<double>();
            return vector<double>(sp.begin(), sp.end());
          },
          nullptr)
      .def_property(
          "shape",
          [](Tensor& t) -> const std::vector<size_t>& { return t.shape(); },
          nullptr)
      .def("__repr__", [](const Tensor& self) {
        std::ostringstream oss;
        oss << self;
        return oss.str();
      });

  py::class_<Variable>(m, "cVariable")
      .def(py::init<Tensor, bool>(), py::arg("data"),
           py::arg("requires_grad") = false)
      .def_property(
          "data", [](const Variable& v) { return v.data(); }, nullptr)
      .def_property(
          "grad", [](const Variable& v) { return v.grad(); }, nullptr)
      .def_property(
          "grad_fn", [](const Variable& v) { return v.grad_fn(); }, nullptr)
      .def_property(
          "requires_grad", [](const Variable& v) { return v.requires_grad(); },
          nullptr)
      .def("__repr__", [](const Variable& self) {
        std::ostringstream oss;
        oss << self;
        return oss.str();
      });

  m.def("add", &add_cust);
  m.def("sub", &sub_cust);
  m.def("mul", &mul_cust);
  m.def("div", &div_cust);
  m.def("exp", &exp_cust);
  m.def("log", &log_cust);
  m.def("sqrt", &sqrt_cust);
  m.def("abs", &abs_cust);
  m.def("sin", &sin_cust);
  m.def("cos", &cos_cust);
  m.def("tan", &tan_cust);
  m.def("clamp", &clamp_cust);
  m.def("matmul", &matmul_cust);
  m.def("transpose", &transpose_cust);
  m.def("sum", &sum_cust);
  m.def("min", &min_cust);
  m.def("max", &max_cust);
}
