#include "tensor_helper.hpp"
#include "autograd/engine/scalar.hpp"

namespace autograd {

Tensor full_like(const Tensor& tensor, Scalar scalar) {
    std::vector<size_t> shape = tensor.shape();
    std::shared_ptr<AbstractBackend> backend = tensor.backend();
    DataType dtype = tensor.type();
    std::vector<Scalar> data(tensor.size(), scalar);
    return Tensor(data, shape, backend, dtype);
}

Tensor ones_like(const Tensor& tensor) {
    return full_like(tensor, 1);
}

Tensor zeros_like(const Tensor& tensor) {
    return full_like(tensor,0);
}

}