#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include "engine.h"

namespace Eigen {
template<>
struct NumTraits<std::shared_ptr<Value>> : NumTraits<float> {
    typedef std::shared_ptr<Value> Real;
    typedef std::shared_ptr<Value> NonInteger;
    typedef std::shared_ptr<Value> Nested;
    
    enum {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 1,
        AddCost = 1,
        MulCost = 1
    };
    static inline Real epsilon() { return std::make_shared<Value>(1e-10); }
    static inline Real dummy_precision() { return std::make_shared<Value>(1e-10); }
    static inline Real Identity() { return std::make_shared<Value>(1.0); }
};
} 

namespace Eigen {
  namespace internal {

    template<>
    struct scalar_sum_op<std::shared_ptr<Value>, std::shared_ptr<Value>> {
      EIGEN_EMPTY_STRUCT_CTOR(scalar_sum_op)
      std::shared_ptr<Value> operator()(const std::shared_ptr<Value>& a, 
                                       const std::shared_ptr<Value>& b) const {
        return a + b;
      }
    };
    template<>
    struct scalar_difference_op<std::shared_ptr<Value>, std::shared_ptr<Value>> {
      EIGEN_EMPTY_STRUCT_CTOR(scalar_difference_op)
      std::shared_ptr<Value> operator()(const std::shared_ptr<Value>& a,
                                        const std::shared_ptr<Value>& b) const {
        return a - b;
      }
    };
    template<>
    struct scalar_product_op<std::shared_ptr<Value>, std::shared_ptr<Value>> {
      EIGEN_EMPTY_STRUCT_CTOR(scalar_product_op)
      std::shared_ptr<Value> operator()(const std::shared_ptr<Value>& a,
                                        const std::shared_ptr<Value>& b) const {
        return a * b;
      }
    };

    template<>
    struct scalar_quotient_op<std::shared_ptr<Value>, std::shared_ptr<Value>> {
      EIGEN_EMPTY_STRUCT_CTOR(scalar_quotient_op)
      std::shared_ptr<Value> operator()(const std::shared_ptr<Value>& a,
                                        const std::shared_ptr<Value>& b) const {
        return a / b; 
      }
    };
    
    // template<>
    // struct scalar_product_op<std::shared_ptr<Value>, float> {
    //   EIGEN_EMPTY_STRUCT_CTOR(scalar_product_op)
    //   std::shared_ptr<Value> operator()(const std::shared_ptr<Value>& a, float b) const {
    //     return a * b;
    //   }
    // };

    // template<>
    // struct scalar_product_op<float, std::shared_ptr<Value>> {
    //   EIGEN_EMPTY_STRUCT_CTOR(scalar_product_op)
    //   std::shared_ptr<Value> operator()(float a, const std::shared_ptr<Value>& b) const {
    //     return a * b;
    //   }
    // };

    template<typename T>
    struct scalar_cast_op<T, std::shared_ptr<Value>> {
      EIGEN_EMPTY_STRUCT_CTOR(scalar_cast_op)
      std::shared_ptr<Value> operator()(const T& scalar) const {
        return std::make_shared<Value>(static_cast<double>(scalar));
      }
    };
    
    // template<typename BinaryOp>
    // struct promote_scalar_arg<BinaryOp, int, std::shared_ptr<Value>> {
    //   typedef std::shared_ptr<Value> ReturnType;
    // };
    
    // template<typename BinaryOp>
    // struct promote_scalar_arg<BinaryOp, float, std::shared_ptr<Value>> {
    //   typedef std::shared_ptr<Value> ReturnType;
    // };
  }
}