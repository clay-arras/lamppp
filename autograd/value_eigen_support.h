#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include "wrapper_engine.h"
#include "engine.h"

namespace Eigen {
template<>
struct NumTraits<SharedValue> : NumTraits<float> {
    typedef SharedValue Real;
    typedef SharedValue NonInteger;
    typedef SharedValue Nested;
    
    enum {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        RequireInitialization = 1,
        ReadCost = 1,
        AddCost = 1,
        MulCost = 1
    };
    static inline Real epsilon() { return SharedValue(1e-10); }
    static inline Real dummy_precision() { return SharedValue(1e-10); }
    static inline Real Identity() { return SharedValue(1.0); }
};
}