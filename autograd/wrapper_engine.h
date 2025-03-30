#ifndef _WRAPPER_ENGINE_H_
#define _WRAPPER_ENGINE_H_

#include "engine.h"
#include <memory>

class SharedValue {
private:
    std::shared_ptr<Value> value;

public:
    SharedValue();
    SharedValue(double data);
    SharedValue(std::shared_ptr<Value> value);

    double getData() const;
    double getGrad() const;
    std::shared_ptr<Value> getPtr() const;

    SharedValue operator+(const SharedValue& other) const;
    SharedValue operator-(const SharedValue& other) const;
    SharedValue operator*(const SharedValue& other) const;
    SharedValue operator/(const SharedValue& other) const;

    SharedValue& operator+=(const SharedValue& other);
    SharedValue& operator-=(const SharedValue& other);
    SharedValue& operator*=(const SharedValue& other);
    SharedValue& operator/=(const SharedValue& other);

    SharedValue operator+(double scalar) const;
    SharedValue operator-(double scalar) const;
    SharedValue operator*(double scalar) const;
    SharedValue operator/(double scalar) const;

    SharedValue& operator+=(double scalar);
    SharedValue& operator-=(double scalar);
    SharedValue& operator*=(double scalar);
    SharedValue& operator/=(double scalar);

    SharedValue exp() const;
    SharedValue log() const;
    SharedValue pow(const SharedValue& exponent) const;
    SharedValue tanh() const;
    SharedValue relu() const;

    void backprop();
    friend std::ostream& operator<<(std::ostream& os, const SharedValue& obj);
};

SharedValue operator+(double scalar, const SharedValue& value);
SharedValue operator-(double scalar, const SharedValue& value);
SharedValue operator*(double scalar, const SharedValue& value);
SharedValue operator/(double scalar, const SharedValue& value);

#endif // _WRAPPER_ENGINE_H_
