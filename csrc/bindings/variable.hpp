#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "flatten.hpp"
#include "lamp3/autograd/core.hpp"
#include "lamp3/common/config.hpp"
#include "lamp3/common/macros.hpp"
#include "lamp3/tensor/core.hpp"
#include "lamp3/tensor/data_type.hpp"

namespace py = pybind11;

using lmp::autograd::Variable;  // NOLINT(google-global-names-in-headers)
using lmp::tensor::DataType;    // NOLINT(google-global-names-in-headers)
using lmp::tensor::DeviceType;  // NOLINT(google-global-names-in-headers)
using lmp::tensor::Tensor;      // NOLINT(google-global-names-in-headers)

#define LMP_VAR_FUNCTION(x) .def(#x, &lmp::autograd::ops::x)

inline void init_variable_overloads(py::class_<Variable>& cls) {
  using lmp::autograd::Variable;
  using lmp::tensor::Scalar;
  namespace ops = lmp::autograd::ops;

  auto wrap = [](const Variable& ref, Scalar s) {
    return Variable(Tensor(std::vector<Scalar>{s}, {1}, ref.data().device(),
                           ref.data().type()));
  };

  cls
      // arithmetic: register (Variable, Variable) FIRST so a Variable operand
      // is never coerced to a scalar.
      .def("__add__", [](const Variable& a, const Variable& b) { return a + b; })
      .def("__add__", [](const Variable& a, Scalar s) { return a + s; })
      .def("__radd__", [](const Variable& a, Scalar s) { return s + a; })
      .def("__sub__", [](const Variable& a, const Variable& b) { return a - b; })
      .def("__sub__", [](const Variable& a, Scalar s) { return a - s; })
      .def("__rsub__", [](const Variable& a, Scalar s) { return s - a; })
      .def("__mul__", [](const Variable& a, const Variable& b) { return a * b; })
      .def("__mul__", [](const Variable& a, Scalar s) { return a * s; })
      .def("__rmul__", [](const Variable& a, Scalar s) { return s * a; })
      .def("__truediv__",
           [](const Variable& a, const Variable& b) { return a / b; })
      .def("__truediv__", [](const Variable& a, Scalar s) { return a / s; })
      .def("__rtruediv__", [](const Variable& a, Scalar s) { return s / a; })
      // pow / matmul: no free operator, dispatch to ops:: directly.
      .def("__pow__",
           [](const Variable& a, const Variable& b) { return ops::pow(a, b); })
      .def("__pow__", [wrap](const Variable& a,
                             Scalar s) { return ops::pow(a, wrap(a, s)); })
      .def("__rpow__", [wrap](const Variable& a,
                              Scalar s) { return ops::pow(wrap(a, s), a); })
      .def("__matmul__", [](const Variable& a,
                            const Variable& b) { return ops::matmul(a, b); })
      .def("__rmatmul__", [](const Variable& a,
                             const Variable& b) { return ops::matmul(b, a); })
      // unary
      .def("__neg__", [](const Variable& a) { return -a; })
      .def("__abs__", &ops::abs)
      // comparisons: (Variable, Variable) first, then (Variable, Scalar).
      // Python maps `s < v` to `v.__gt__(s)`, so no reflected dunders needed.
      .def("__eq__", [](const Variable& a, const Variable& b) { return a == b; })
      .def("__eq__", [](const Variable& a, Scalar s) { return a == s; })
      .def("__ne__", [](const Variable& a, const Variable& b) { return a != b; })
      .def("__ne__", [](const Variable& a, Scalar s) { return a != s; })
      .def("__ge__", [](const Variable& a, const Variable& b) { return a >= b; })
      .def("__ge__", [](const Variable& a, Scalar s) { return a >= s; })
      .def("__le__", [](const Variable& a, const Variable& b) { return a <= b; })
      .def("__le__", [](const Variable& a, Scalar s) { return a <= s; })
      .def("__gt__", [](const Variable& a, const Variable& b) { return a > b; })
      .def("__gt__", [](const Variable& a, Scalar s) { return a > s; })
      .def("__lt__", [](const Variable& a, const Variable& b) { return a < b; })
      .def("__lt__", [](const Variable& a, Scalar s) { return a < s; })
          LMP_FOR_EACH_CARTESIAN_PRODUCT(
              LMP_VAR_FUNCTION,
              (to, expand_dims, squeeze, reshape, exp, log, sqrt, abs, sin, cos,
               tan, clamp, sum, max, min, prod))
              .def("T", &lmp::autograd::ops::transpose);
}

#undef LMP_VAR_FUNCTION

inline void init_variable(py::module_& m) {
  auto cls =
      py::class_<Variable>(m, "_Variable")
          .def(py::init<Tensor, bool>(), py::arg("data"),
               py::arg("requires_grad"))
          .def(py::init([](const py::object& data, bool requires_grad,
                           DeviceType device, DataType dtype) {
                 auto [flat, shape] = lmp::bindings::flatten_array(data);
                 return Variable(Tensor(flat, shape, device, dtype),
                                 requires_grad);
               }),
               py::arg("data"), py::arg("requires_grad") = false,
               py::arg("device") = lmp::DEFAULT_DEVICE,
               py::arg("dtype") = lmp::DEFAULT_DTYPE)
          .def_property("data", &Variable::data, nullptr)
          .def_property("grad", &Variable::grad, nullptr)
          .def_property("grad_fn", &Variable::grad_fn, nullptr)
          .def_property("requires_grad", &Variable::requires_grad, nullptr)
          .def("backward", &Variable::backward)
          .def("tolist",
               [](const Variable& self) {
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