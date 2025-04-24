#pragma once

#ifndef EIGEN_BACKEND_H
#define EIGEN_BACKEND_H

#include "autograd/engine/backend.h"
#include <Eigen/Core>

namespace autograd {

struct EigenBackend : AbstractBackend {
    TensorImpl add(const TensorImpl& a, const TensorImpl& b) override;
    TensorImpl sub(const TensorImpl& a, const TensorImpl& b) override;
    TensorImpl mul(const TensorImpl& a, const TensorImpl& b) override;
    TensorImpl div(const TensorImpl& a, const TensorImpl& b) override;

    TensorImpl log(const TensorImpl& a) override;
    TensorImpl exp(const TensorImpl& a) override;
    TensorImpl relu(const TensorImpl& a) override;

    TensorImpl matmul(const TensorImpl& a, const TensorImpl& b) override;
    TensorImpl transpose(const TensorImpl& a) override;

    TensorImpl sum(const TensorImpl& a, int axis) override;
    // TensorImpl mean(const TensorImpl& a, int axis) override;
    // TensorImpl max(const TensorImpl& a, int axis) override;
    // TensorImpl min(const TensorImpl& a, int axis) override;

    static Eigen::Map<Eigen::MatrixXf> as_matrix(const TensorImpl& impl, int rows, int cols) {
        return Eigen::Map<Eigen::MatrixXf>(const_cast<float*>(impl.data.data()), rows, cols);
    }

    static Eigen::Map<Eigen::ArrayXXf> as_array(const TensorImpl& impl) {
        return Eigen::Map<Eigen::ArrayXXf>(const_cast<float*>(impl.data.data()),
                                        impl.data.size(), 1);
    }
};

} // namespace autograd

#endif // EIGEN_BACKEND_H