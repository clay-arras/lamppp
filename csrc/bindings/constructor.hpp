#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "flatten.hpp"
#include "lamp3/tensor/core.hpp"
#include "lamp3/autograd/core.hpp"
#include "lamp3/common/config.hpp"
#include "lamp3/common/macros.hpp"
#include "lamp3/tensor/data_type.hpp"

namespace py = pybind11;

inline void init_constructor(py::module_& m) {
    m.def("zeros", &lmp::autograd::zeros);
    m.def("ones", &lmp::autograd::ones);
    m.def("rand", &lmp::autograd::rand);
    m.def("randn", &lmp::autograd::randn);
    m.def(
        "tensor",
        [](const py::object& data, bool requires_grad,
           lmp::tensor::DeviceType device, lmp::tensor::DataType dtype) {
          auto [flat, shape] = lmp::bindings::flatten_array(data);
          return lmp::autograd::Variable(
              lmp::tensor::Tensor(flat, shape, device, dtype), requires_grad);
        },
        py::arg("data"), py::arg("requires_grad") = false,
        py::arg("device") = lmp::DEFAULT_DEVICE,
        py::arg("dtype") = lmp::DEFAULT_DTYPE);
}