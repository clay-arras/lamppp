#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lamppp/autograd/core.hpp"
#include "lamppp/tensor/core.hpp"

namespace py = pybind11;

using lmp::tensor::DataType;

void init_data_type(py::module_& m) {
  py::enum_<DataType>(m, "dtype")
      .value("float32", DataType::Float32)
      .value("float64", DataType::Float64)
      .value("bool", DataType::Bool)
      .value("int16", DataType::Int16)
      .value("int32", DataType::Int32)
      .value("int64", DataType::Int64)
      .export_values();
}