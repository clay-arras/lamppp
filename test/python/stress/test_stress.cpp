#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "autograd/engine/tensor.h"
#include "autograd/engine/function.h"

namespace py = pybind11;

using autograd::Tensor;
using autograd::Variable;

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

Variable relu_cust(const Variable& a) {
  Variable c = a.relu();
  c.backward();
  return c;
}

Variable exp_cust(const Variable& a) {
  Variable c = a.exp();
  c.backward();
  return c;
}

Variable log_cust(const Variable& a) {
  Variable c = a.log();
  c.backward();
  return c;
}

Variable matmul_cust(const Variable& a, const Variable& b) {
  Variable c = a.matmul(b);
  c.backward();
  return c;
}

Variable transpose_cust(const Variable& a) {
  Variable c = a.transpose();
  c.backward();
  return c;
}

Variable sum_cust(const Variable& a, int axis) {
  Variable c = a.sum(axis);
  c.backward();
  return c;
}

}  // namespace

PYBIND11_MODULE(cpp_custom_bind, m) {
  py::class_<Tensor>(m, "cTensor")
      .def(py::init<std::vector<float>, std::vector<int>>(), py::arg("data"),
           py::arg("shape"))
      .def_property(
          "data", [](Tensor& t) -> std::vector<float>& { return t.data(); },
          [](Tensor& t, const std::vector<float>& d) { t.data() = d; })
      .def_property(
          "shape", [](Tensor& t) -> std::vector<int>& { return t.shape(); },
          [](Tensor& t, const std::vector<int>& s) { t.shape() = s; });

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
          nullptr);

  m.def("add", &add_cust);
  m.def("sub", &sub_cust);
  m.def("mul", &mul_cust);
  m.def("div", &div_cust);
  m.def("relu", &relu_cust);
  m.def("exp", &exp_cust);
  m.def("log", &log_cust);
  m.def("matmul", &matmul_cust);
  m.def("transpose", &transpose_cust);
  m.def("sum", &sum_cust);
}
