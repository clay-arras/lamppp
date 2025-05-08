// #include <iostream>
// #include <memory>

// class TensorImpl;

// class AbstractBackend {
//  public:
//   virtual void foo(TensorImpl a) = 0;
// };

// class TensorImpl {
//  public:
//   TensorImpl(std::shared_ptr<AbstractBackend> impl) : impl(impl) {};

//   static void bar(TensorImpl a) { a.impl->foo(a); }

//   std::shared_ptr<AbstractBackend> impl;
// };

// class CPUBackend : public AbstractBackend {
//  public:
//   void foo(TensorImpl a) override {
//     std::cout << "CPUBackend foo" << std::endl;
//   };
// };

// int main() {
//   TensorImpl a = TensorImpl(std::make_shared<CPUBackend>());
//   TensorImpl::bar(a);

//   return 0;
// }

// #include <boost/preprocessor/seq/elem.hpp>
// #include <boost/preprocessor/seq/for_each_product.hpp>
// #include <iostream>

// template <typename U, typename V, typename T>
// struct MyTemplate {
//   void printTypes() {
//     std::cout << "Types: " << typeid(U).name() << ", " << typeid(V).name()
//               << ", " << typeid(T).name() << std::endl;
//   }
// };

// #define TYPES (int)(float)(double)

// #define INSTANTIATE(r, product)                                     \
//   template struct MyTemplate<BOOST_PP_SEQ_ELEM(0, product), /* U */ \
//                              BOOST_PP_SEQ_ELEM(1, product), /* V */ \
//                              BOOST_PP_SEQ_ELEM(2, product)  /* T */ \
//                              >;

// BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE, (TYPES)(TYPES)(TYPES))

// #undef INSTANTIATE
// #undef TYPES

#include "autograd/engine/data_type.hpp"
#include "autograd/engine/device_type.hpp"
#include "autograd/engine/native/empty.cuh"
#include "autograd/engine/tensor.hpp"
#include "autograd/engine/tensor_helper.hpp"
#include "autograd/engine/variable.hpp"
#include "autograd/engine/variable_ops.hpp"

int main() {
  std::vector<autograd::Scalar> data1 = {1.0, 2.0, -1.0};
  std::vector<size_t> shape1 = {1, 3};
  autograd::Tensor tensor_data1 =
      autograd::Tensor(data1, shape1, DeviceType::CUDA, DataType::Float32);

  // std::vector<autograd::Scalar> data2 = {1.0f, 2.0f, 3.0f};
  // std::vector<size_t> shape2 = {1, 3};
  // autograd::Tensor tensor_data2 =
  //     autograd::Tensor(data2, shape2, DeviceType::CUDA, DataType::Float32);

  // autograd::Variable variable_data1(tensor_data1, true);
  // autograd::Variable variable_data2(tensor_data2, true);

  std::cout << tensor_data1 << std::endl;
  // autograd::Variable result = variable_data1 + variable_data2;

  // std::cout << "Variable 1: " << variable_data1 << std::endl;
  // std::cout << "Variable 2: " << variable_data2 << std::endl;
  // std::cout << "Result: " << result << std::endl;

  // result.backward();

  // std::cout << "Variable 1: " << variable_data1 << std::endl;
  // std::cout << "Variable 2: " << variable_data2 << std::endl;
  // std::cout << "Result: " << result << std::endl;

  // std::vector<float> host_data = {1.0f, 2.0f, -1.0f};
  // std::cout << "Original data: ";
  // for (float x : host_data) {
  //   std::cout << x << " ";
  // }
  // std::cout << std::endl;

  // size_t all_size = host_data.size() * sizeof(float);

  // autograd::DataPtr device_ptr =
  //     autograd::empty_stub(DeviceType::CUDA, all_size);
  // autograd::copy_stub(DeviceType::CPU, host_data.data(), device_ptr.data,
  //                     all_size, DeviceType::CUDA);

  // std::vector<float> result_data(host_data.size());
  // autograd::copy_stub(DeviceType::CUDA, device_ptr.data, result_data.data(),
  //                     all_size, DeviceType::CPU);

  // std::cout << "Data after round trip through CUDA: ";
  // for (float x : result_data) {
  //   std::cout << x << " ";
  // }
  // std::cout << std::endl;
}