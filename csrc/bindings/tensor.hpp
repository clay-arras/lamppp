#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lamppp/tensor/core.hpp"

namespace py = pybind11;

using lmp::tensor::DeviceType;
using lmp::tensor::DataType;
using lmp::tensor::DeviceType;
using lmp::tensor::Tensor;

void init_tensor(py::module_& m) {
  py::class_<Tensor>(m, "_Tensor")
      .def(py::init<const std::vector<double>, const std::vector<size_t>,
                    DeviceType, DataType>(),
           py::arg("data"), py::arg("shape"), py::arg("device"),
           py::arg("dtype"))
      .def_property("shape", &Tensor::shape, nullptr)
      .def_property("device", &Tensor::device, nullptr)
      .def_property("dtype", &Tensor::type, nullptr)
      .def("index", &Tensor::index, py::arg("idx"))
      .def("reshape", &Tensor::reshape, py::arg("new_shape"))
      .def("squeeze", &Tensor::squeeze, py::arg("dim"))
      .def("expand_dims", &Tensor::expand_dims, py::arg("dim"))
      .def("to", &Tensor::to, py::arg("device"))
      .def("copy", &Tensor::copy, py::arg("other"))
      .def("fill", &Tensor::fill, py::arg("item"))
      .def(
          "tolist",
          [](Tensor& t) -> std::vector<double> { return t.to_vector<double>(); })
      .def("__repr__", [](const Tensor& self) {
        std::ostringstream oss;
        oss << self;
        return oss.str();
      });
}