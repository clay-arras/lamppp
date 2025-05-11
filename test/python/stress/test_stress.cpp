#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

#include "autograd/engine/data_type.hpp"
#include "autograd/engine/function.hpp"
#include "autograd/engine/tensor.hpp"
#include "autograd/engine/variable_ops.hpp"

namespace py = pybind11;

using autograd::Tensor;
using autograd::Variable;
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

Variable relu_cust(const Variable& a) {
  Variable c = autograd::relu(a);
  c.backward();
  return c;
}

Variable exp_cust(const Variable& a) {
  Variable c = autograd::exp(a);
  c.backward();
  return c;
}

Variable log_cust(const Variable& a) {
  Variable c = autograd::log(a);
  c.backward();
  return c;
}

Variable matmul_cust(const Variable& a, const Variable& b) {
  Variable c = autograd::matmul(a, b);
  c.backward();
  return c;
}

Variable transpose_cust(const Variable& a) {
  Variable c = autograd::transpose(a);
  c.backward();
  return c;
}

Variable sum_cust(const Variable& a, int axis) {
  Variable c = autograd::sum(a, axis);
  c.backward();
  return c;
}

// Tensor equivalents
Tensor add_tensor(const Tensor& a, const Tensor& b) {
  return a + b;
}

Tensor sub_tensor(const Tensor& a, const Tensor& b) {
  return a - b;
}

Tensor mul_tensor(const Tensor& a, const Tensor& b) {
  return a * b;
}

Tensor div_tensor(const Tensor& a, const Tensor& b) {
  return a / b;
}

}  // namespace

PYBIND11_MODULE(cpp_custom_bind, m) {
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

  // Variable operations
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

  // Tensor operations
  m.def("add_tensor", &add_tensor);
  m.def("sub_tensor", &sub_tensor);
  m.def("mul_tensor", &mul_tensor);
  m.def("div_tensor", &div_tensor);
}
