// // Car.hpp
// #include <memory>
// enum class DataType { i32, f32, f64 };

// class Car {
// public:
//   static Car make(DataType t);            // factory

//   /* generic API */
//   int    wheels() const;                  // never mentions T
//   double mass()   const;

//   template<class T>                      // opt-in API
//   const T* data() const;                 // similar to fl::Tensor::data<T>()

// private:
//   struct ImplBase {
//     virtual ~ImplBase() = default;
//     virtual int    wheels() const = 0;
//     virtual double mass()   const = 0;
//   };

//   template<class T> struct Impl;          // see .cpp
//   std::unique_ptr<ImplBase> p_;
// };

// // Car.cpp
// template<class T>
// struct Car::Impl : ImplBase {
//   std::vector<T> buf_;
//   int wheels() const override { return 4; }
//   double mass() const override { return buf_.size()*sizeof(T); }
// };

// Car Car::make(DataType d) {
//   switch (d) {
//     case DataType::i32: return Car{std::make_unique<Impl<int>>()};
//     case DataType::f32: return Car{std::make_unique<Impl<float>>()};
//     default:            return Car{std::make_unique<Impl<double>>()};
//   }
// }

// template<class T>
// const T* Car::data() const {
//   if (auto m = dynamic_cast<Model<T>*>(p_.get()))
//     return &m->impl_.payload();
//   return nullptr;                         // or throw
// }

#include <vector>
// #include "autograd/autograd_umbrella.h"
#include "autograd/engine/backend/cuda_backend.h"
#include "autograd/engine/tensor.h"

int main() {
  // Create two tensors
  std::vector<float> data1 = {1.0f, 2.0f, 3.0f};
  std::vector<int> shape1 = {3};
  autograd::Tensor tensor1 =
      autograd::Tensor::create<float, autograd::CudaBackend<float>>(data1,
                                                                    shape1);

  std::vector<float> data2 = {4.0f, 5.0f, 6.0f};
  std::vector<int> shape2 = {3};
  autograd::Tensor tensor2 =
      autograd::Tensor::create<float, autograd::CudaBackend<float>>(data2,
                                                                    shape2);

  autograd::Tensor result = tensor1 + tensor2;

  std::cout << "Result of tensor addition: " << result << std::endl;

  return 0;
}
