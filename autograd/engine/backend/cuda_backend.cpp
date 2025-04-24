#include "cuda_backend.h"

namespace autograd {
TensorImpl CudaBackend::add(const TensorImpl& a, const TensorImpl& b) {
    return TensorImpl();
}

TensorImpl CudaBackend::sub(const TensorImpl& a, const TensorImpl& b) {
    return TensorImpl();
}

TensorImpl CudaBackend::mul(const TensorImpl& a, const TensorImpl& b) {
    return TensorImpl();
}

TensorImpl CudaBackend::div(const TensorImpl& a, const TensorImpl& b) {
    return TensorImpl();
}

TensorImpl CudaBackend::log(const TensorImpl& a) {
    return TensorImpl();
}

TensorImpl CudaBackend::exp(const TensorImpl& a) {
    return TensorImpl();
}

TensorImpl CudaBackend::relu(const TensorImpl& a) {
    return TensorImpl();
}

TensorImpl CudaBackend::matmul(const TensorImpl& a, const TensorImpl& b) {
    return TensorImpl();
}

TensorImpl CudaBackend::transpose(const TensorImpl& a) {
    return TensorImpl();
}

TensorImpl CudaBackend::equal(const TensorImpl& a, const TensorImpl& b) {
    return TensorImpl();
}

TensorImpl CudaBackend::not_equal(const TensorImpl& a, const TensorImpl& b) {
    return TensorImpl();
}

TensorImpl CudaBackend::greater_equal(const TensorImpl& a, const TensorImpl& b) {
    return TensorImpl();
}

TensorImpl CudaBackend::less_equal(const TensorImpl& a, const TensorImpl& b) {
    return TensorImpl();
}

TensorImpl CudaBackend::greater_than(const TensorImpl& a, const TensorImpl& b) {
    return TensorImpl();
}

TensorImpl CudaBackend::less_than(const TensorImpl& a, const TensorImpl& b) {
    return TensorImpl();
}

TensorImpl CudaBackend::sum(const TensorImpl& a, int axis) {
    return TensorImpl();
}

}