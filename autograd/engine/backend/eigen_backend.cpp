// #include "eigen_backend.h"
// #include <numeric>

// namespace autograd {

// std::shared_ptr<TensorImpl> EigenBackend::add(const TensorImpl& a, const TensorImpl& b) {
//   assert(a.shape == b.shape);
//   int sz =
//       std::accumulate(a.shape.begin(), a.shape.end(), 1, std::multiplies<>());
//   std::vector<float> res_data(sz);
//   Eigen::Map<Eigen::ArrayXXf> res(res_data.data(), sz, 1);
//   res = as_array(a) + as_array(b);
//   return make_shared<TensorImplModel>(res_data, a.shape);
// }

// std::shared_ptr<TensorImpl> EigenBackend::sub(const TensorImpl& a, const TensorImpl& b) {
//   assert(a.shape == b.shape);
//   int sz =
//       std::accumulate(a.shape.begin(), a.shape.end(), 1, std::multiplies<>());
//   std::vector<float> res_data(sz);
//   Eigen::Map<Eigen::ArrayXXf> res(res_data.data(), sz, 1);
//   res = as_array(a) - as_array(b);
//   return std::make_shared<TensorImplModel<float>>(res_data, a.shape);
// }

// std::shared_ptr<TensorImpl> EigenBackend::mul(const TensorImpl& a, const TensorImpl& b) {
//   assert(a.shape == b.shape);
//   int sz =
//       std::accumulate(a.shape.begin(), a.shape.end(), 1, std::multiplies<>());
//   std::vector<float> res_data(sz);
//   Eigen::Map<Eigen::ArrayXXf> res(res_data.data(), sz, 1);
//   res = as_array(a) * as_array(b);
//   return std::make_shared<TensorImplModel>(res_data, a.shape);
// }

// std::shared_ptr<TensorImpl> EigenBackend::div(const TensorImpl& a, const TensorImpl& b) {
//   assert(a.shape == b.shape);
//   int sz =
//       std::accumulate(a.shape.begin(), a.shape.end(), 1, std::multiplies<>());
//   std::vector<float> res_data(sz);
//   Eigen::Map<Eigen::ArrayXXf> res(res_data.data(), sz, 1);
//   res = as_array(a) / as_array(b);
//   return std::make_shared<TensorImplModel>(res_data, a.shape);
// }

// std::shared_ptr<TensorImpl> EigenBackend::equal(const TensorImpl& a, const TensorImpl& b) {
//   assert(a.shape == b.shape);
//   int sz =
//       std::accumulate(a.shape().begin(), a.shape.end(), 1, std::multiplies<>());
//   std::vector<float> res_data(sz);
//   Eigen::Map<Eigen::ArrayXXf> res(res_data.data(), sz, 1);
//   res = (as_array(a) == as_array(b)).cast<float>();
//   return std::make_shared<TensorImplModel>(res_data, a.shape);
// }

// std::shared_ptr<TensorImpl> EigenBackend::not_equal(const TensorImpl& a, const TensorImpl& b) {
//   assert(a.shape == b.shape);
//   int sz =
//       std::accumulate(a.shape.begin(), a.shape.end(), 1, std::multiplies<>());
//   std::vector<float> res_data(sz);
//   Eigen::Map<Eigen::ArrayXXf> res(res_data.data(), sz, 1);
//   res = (as_array(a) != as_array(b)).cast<float>();
//   return std::make_shared<TensorImplModel>(res_data, a.shape);
// }

// std::shared_ptr<TensorImpl> EigenBackend::greater_equal(const TensorImpl& a,
//                                        const TensorImpl& b) {
//   assert(a.shape == b.shape);
//   int sz =
//       std::accumulate(a.shape.begin(), a.shape.end(), 1, std::multiplies<>());
//   std::vector<float> res_data(sz);
//   Eigen::Map<Eigen::ArrayXXf> res(res_data.data(), sz, 1);
//   res = (as_array(a) >= as_array(b)).cast<float>();
//   return std::make_shared<TensorImplModel>(res_data, a.shape);
// }

// std::shared_ptr<TensorImpl> EigenBackend::less_equal(const TensorImpl& a, const TensorImpl& b) {
//   assert(a.shape == b.shape);
//   int sz =
//       std::accumulate(a.shape.begin(), a.shape.end(), 1, std::multiplies<>());
//   std::vector<float> res_data(sz);
//   Eigen::Map<Eigen::ArrayXXf> res(res_data.data(), sz, 1);
//   res = (as_array(a) <= as_array(b)).cast<float>();
//   return std::make_shared<TensorImplModel>(res_data, a.shape);
// }

// std::shared_ptr<TensorImpl> EigenBackend::greater_than(const TensorImpl& a,
//                                       const TensorImpl& b) {
//   assert(a.shape == b.shape);
//   int sz =
//       std::accumulate(a.shape.begin(), a.shape.end(), 1, std::multiplies<>());
//   std::vector<float> res_data(sz);
//   Eigen::Map<Eigen::ArrayXXf> res(res_data.data(), sz, 1);
//   res = (as_array(a) > as_array(b)).cast<float>();
//   return std::make_shared<TensorImplModel>(res_data, a.shape);
// }

// std::shared_ptr<TensorImpl> EigenBackend::less_than(const TensorImpl& a, const TensorImpl& b) {
//   assert(a.shape == b.shape);
//   int sz =
//       std::accumulate(a.shape.begin(), a.shape.end(), 1, std::multiplies<>());
//   std::vector<float> res_data(sz);
//   Eigen::Map<Eigen::ArrayXXf> res(res_data.data(), sz, 1);
//   res = (as_array(a) < as_array(b)).cast<float>();
//   return std::make_shared<TensorImplModel>(res_data, a.shape);
// }

// std::shared_ptr<TensorImpl> EigenBackend::log(const TensorImpl& a) {
//   int sz =
//       std::accumulate(a.shape.begin(), a.shape.end(), 1, std::multiplies<>());
//   std::vector<float> res_data(sz);
//   Eigen::Map<Eigen::ArrayXXf> res(res_data.data(), sz, 1);
//   res = as_array(a).log();
//   return std::make_shared<TensorImplModel>(res_data, a.shape);
// }

// std::shared_ptr<TensorImpl> EigenBackend::exp(const TensorImpl& a) {
//   int sz =
//       std::accumulate(a.shape.begin(), a.shape.end(), 1, std::multiplies<>());
//   std::vector<float> res_data(sz);
//   Eigen::Map<Eigen::ArrayXXf> res(res_data.data(), sz, 1);
//   res = as_array(a).exp();
//   return std::make_shared<TensorImplModel>(res_data, a.shape);
// }

// std::shared_ptr<TensorImpl> EigenBackend::relu(const TensorImpl& a) {
//   int sz =
//       std::accumulate(a.shape.begin(), a.shape.end(), 1, std::multiplies<>());
//   std::vector<float> res_data(sz);
//   Eigen::Map<Eigen::ArrayXXf> res(res_data.data(), sz, 1);
//   res = as_array(a).max(0);
//   return std::make_shared<TensorImplModel>(res_data, a.shape);
// }

// std::shared_ptr<TensorImpl> EigenBackend::matmul(const TensorImpl& a, const TensorImpl& b) {
//   assert(a.shape.size() == 2 && b.shape.size() == 2);
//   assert(a.shape[1] == b.shape[0]);

//   std::vector<int> out_shape = {a.shape[0], b.shape[1]};
//   int sz = a.shape[0] * b.shape[1];
//   std::vector<float> res_data(sz);

//   Eigen::Map<Eigen::MatrixXf> res(res_data.data(), a.shape[0], b.shape[1]);
//   res = as_matrix(a, a.shape[0], a.shape[1]) *
//         as_matrix(b, b.shape[0], b.shape[1]);

//   return std::make_shared<TensorImplModel>(res_data, out_shape);
// }

// std::shared_ptr<TensorImpl> EigenBackend::transpose(const TensorImpl& a) {
//   assert(a.shape.size() == 2);

//   std::vector<int> out_shape = {a.shape[1], a.shape[0]};
//   int sz = a.shape[0] * a.shape[1];
//   std::vector<float> res_data(sz);

//   Eigen::Map<Eigen::MatrixXf> res(res_data.data(), a.shape[1], a.shape[0]);
//   res = as_matrix(a, a.shape[0], a.shape[1]).transpose();

//   return std::make_shared<TensorImplModel>(res_data, out_shape);
// }

// std::shared_ptr<TensorImpl> EigenBackend::sum(const TensorImpl& a, int axis) {
//   assert(axis >= 0 && axis < static_cast<int>(a.shape.size()));
//   std::vector<int> new_shape = a.shape;
//   new_shape[axis] = 1;

//   std::vector<float> res_data(a.data.size() / a.shape[axis]);

//   if (axis == 0) {
//     Eigen::Map<Eigen::Matrix<float, 1, Eigen::Dynamic>> res(res_data.data(), 1,
//                                                             a.shape[1]);
//     res = as_array(a).reshaped(a.shape[0], a.shape[1]).colwise().sum();
//   } else if (axis == 1) {
//     Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>> res(res_data.data(),
//                                                             a.shape[0], 1);
//     res = as_array(a).reshaped(a.shape[0], a.shape[1]).rowwise().sum();
//   } else {
//     assert(0);  // Only supporting 2D tensors for now
//   }

//   return std::make_shared<TensorImplModel>(res_data, new_shape);
// }

// std::shared_ptr<TensorImpl> EigenBackend::max(const TensorImpl& a, int axis) {
//   assert(axis >= 0 && axis < static_cast<int>(a.shape.size()));
//   std::vector<int> new_shape = a.shape;
//   new_shape[axis] = 1;

//   std::vector<float> res_data(a.data.size() / a.shape[axis]);

//   if (axis == 0) {
//     Eigen::Map<Eigen::Matrix<float, 1, Eigen::Dynamic>> res(res_data.data(), 1,
//                                                             a.shape[1]);
//     res = as_array(a).reshaped(a.shape[0], a.shape[1]).colwise().maxCoeff();
//   } else if (axis == 1) {
//     Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>> res(res_data.data(),
//                                                             a.shape[0], 1);
//     res = as_array(a).reshaped(a.shape[0], a.shape[1]).rowwise().maxCoeff();
//   } else {
//     assert(0);
//   }

//   return std::make_shared<TensorImplModel>(res_data, new_shape);
// }

// }  // namespace autograd