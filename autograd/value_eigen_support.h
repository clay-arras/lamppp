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
};
} 