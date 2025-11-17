#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lamppp/autograd/core.hpp"
#include "lamppp/tensor/core.hpp"

namespace py = pybind11;

inline void init_data_type(py::module_& m) {
  py::enum_<lmp::tensor::DataType>(m, "dtype")
      .value("float32", lmp::tensor::DataType::Float32)
      .value("float64", lmp::tensor::DataType::Float64)
      .value("bool", lmp::tensor::DataType::Bool)
      .value("int16", lmp::tensor::DataType::Int16)
      .value("int32", lmp::tensor::DataType::Int32)
      .value("int64", lmp::tensor::DataType::Int64)
      .export_values();
}