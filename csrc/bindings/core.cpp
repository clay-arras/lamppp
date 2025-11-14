#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lamppp/autograd/core.hpp"
#include "lamppp/tensor/core.hpp"

#include "data_type.hpp"
#include "device_type.hpp"
#include "tensor.hpp"
#include "variable.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pylamp, m) {
    init_data_type(m);
    init_device_type(m);
    init_tensor(m);
    init_variable(m);
}
