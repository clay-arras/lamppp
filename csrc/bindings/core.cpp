#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "lamppp/autograd/core.hpp"
#include "lamppp/tensor/core.hpp"

#include "data_type.hpp"
#include "device_type.hpp"
#include "tensor.hpp"
#include "variable.hpp"
#include "functions/expand.hpp"
#include "functions/binary.hpp"
#include "functions/unary.hpp"
#include "functions/matrix.hpp"
#include "functions/reduct.hpp"
#include "functions/view.hpp"
#include "functions/conv.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pylamp, m) {
    init_data_type(m);
    init_device_type(m);
    init_tensor(m);
    init_variable(m);

    init_expand(m);
    init_binary(m);
    init_unary(m);
    init_matrix(m);
    init_reduct(m);
    init_view(m);
    init_conv(m);
}
