// #pragma once

// #ifndef EIGEN_BACKEND_H
// #define EIGEN_BACKEND_H

// #include <memory>
// #include <Eigen/Core>
// #include "include/lamppp/tensor/backend.h"

// namespace autograd {

// struct EigenBackend : AbstractBackend {
//   std::shared_ptr<TensorImpl> add(const TensorImpl& a, const TensorImpl& b) override;
//   std::shared_ptr<TensorImpl> sub(const TensorImpl& a, const TensorImpl& b) override;
//   std::shared_ptr<TensorImpl> mul(const TensorImpl& a, const TensorImpl& b) override;
//   std::shared_ptr<TensorImpl> div(const TensorImpl& a, const TensorImpl& b) override;

//   std::shared_ptr<TensorImpl> log(const TensorImpl& a) override;
//   std::shared_ptr<TensorImpl> exp(const TensorImpl& a) override;
//   std::shared_ptr<TensorImpl> relu(const TensorImpl& a) override;

//   std::shared_ptr<TensorImpl> matmul(const TensorImpl& a, const TensorImpl& b) override;
//   std::shared_ptr<TensorImpl> transpose(const TensorImpl& a) override;

//   std::shared_ptr<TensorImpl> equal(const TensorImpl& a, const TensorImpl& b) override;
//   std::shared_ptr<TensorImpl> not_equal(const TensorImpl& a, const TensorImpl& b) override;
//   std::shared_ptr<TensorImpl> greater_equal(const TensorImpl& a, const TensorImpl& b) override;
//   std::shared_ptr<TensorImpl> less_equal(const TensorImpl& a, const TensorImpl& b) override;
//   std::shared_ptr<TensorImpl> greater_than(const TensorImpl& a, const TensorImpl& b) override;
//   std::shared_ptr<TensorImpl> less_than(const TensorImpl& a, const TensorImpl& b) override;

//   std::shared_ptr<TensorImpl> sum(const TensorImpl& a, size_t axis) override;
//   std::shared_ptr<TensorImpl> max(const TensorImpl& a, size_t axis) override;
//   // TensorImpl mean(const TensorImpl& a, size_t axis) override;
//   // TensorImpl min(const TensorImpl& a, size_t axis) override;

//   static Eigen::Map<Eigen::MatrixXf> as_matrix(const TensorImpl& impl, size_t rows,
//                                                size_t cols) {
//     return Eigen::Map<Eigen::MatrixXf>(const_cast<float*>(impl.data_ptr()),
//                                        rows, cols);
//   }

//   static Eigen::Map<Eigen::ArrayXXf> as_array(const TensorImpl& impl) {
//     return Eigen::Map<Eigen::ArrayXXf>(const_cast<float*>(impl.data_ptr()),
//                                        impl.data_size(), 1);
//   }
// };

// }  // namespace autograd

// #endif  // EIGEN_BACKEND_H