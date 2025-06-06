#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

#include "lamppp/autograd/core.hpp"
#include "lamppp/tensor/core.hpp"

namespace py = pybind11;

using lmp::autograd::Variable;
using lmp::tensor::DeviceType;
using lmp::tensor::DataType;
using lmp::tensor::Tensor;
using std::ostringstream;
using std::vector;

namespace {

Variable add_cust(const Variable& a, const Variable& b) {
  return a + b;
}

Variable sub_cust(const Variable& a, const Variable& b) {
  return a - b;
}

Variable mul_cust(const Variable& a, const Variable& b) {
  return a * b;
}

Variable div_cust(const Variable& a, const Variable& b) {
  return a / b;
}

Variable exp_cust(const Variable& a) {
  return lmp::autograd::ops::exp(a);
}

Variable log_cust(const Variable& a) {
  return lmp::autograd::ops::log(a);
}

Variable sqrt_cust(const Variable& a) {
  return lmp::autograd::ops::sqrt(a);
}

Variable abs_cust(const Variable& a) {
  return lmp::autograd::ops::abs(a);
}

Variable sin_cust(const Variable& a) {
  return lmp::autograd::ops::sin(a);
}

Variable cos_cust(const Variable& a) {
  return lmp::autograd::ops::cos(a);
}

Variable tan_cust(const Variable& a) {
  return lmp::autograd::ops::tan(a);
}

Variable clamp_cust(const Variable& a, double min_val, double max_val) {
  return lmp::autograd::ops::clamp(a, min_val, max_val);
}

Variable matmul_cust(const Variable& a, const Variable& b) {
  return lmp::autograd::ops::matmul(a, b);
}

Variable transpose_cust(const Variable& a) {
  return lmp::autograd::ops::transpose(a);
}

Variable sum_cust(const Variable& a, size_t axis) {
  return lmp::autograd::ops::sum(a, axis);
}

Variable max_cust(const Variable& a, size_t axis) {
  return lmp::autograd::ops::max(a, axis);
}

Variable min_cust(const Variable& a, size_t axis) {
  return lmp::autograd::ops::min(a, axis);
}

Variable reshape_cust(const Variable& a, const std::vector<size_t>& shape) {
  return lmp::autograd::ops::reshape(a, shape);
}

Variable expand_cust(const Variable& a, size_t axis) {
  return lmp::autograd::ops::expand_dims(a, axis);
}

Variable squeeze_cust(const Variable& a, size_t axis) {
  return lmp::autograd::ops::squeeze(a, axis);
}

}  // namespace

PYBIND11_MODULE(lamppp_module, m) {
  py::enum_<DataType>(m, "cDataType")
      .value("Float32", DataType::Float32)
      .value("Float64", DataType::Float64)
      .value("Int32", DataType::Int32);

  py::enum_<DeviceType>(m, "cDeviceType")
      .value("CPU", DeviceType::CPU)
      .value("CUDA", DeviceType::CUDA);

  py::class_<Tensor>(m, "cTensor")
      .def(py::init<const std::vector<double>, const std::vector<size_t>, DeviceType, DataType> (),
           py::arg("data"), py::arg("shape"), py::arg("device"), py::arg("dtype"))
      .def_property(
          "data",
          [](Tensor& t) -> vector<double> { return t.to_vector<double>(); },
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
      .def("backward", &Variable::backward)
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
  m.def("squeeze", &squeeze_cust);
  m.def("expand_dims", &expand_cust);
  m.def("reshape", &reshape_cust);
}
