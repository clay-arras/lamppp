#ifndef _ENGINE_H_
#define _ENGINE_H_

#include <functional>
#include <unordered_set>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <memory>
#include <iostream>

class Value : public std::enable_shared_from_this<Value> {
private:
    std::function<void()> backward; 
    std::unordered_set<std::shared_ptr<Value>> prev;
    std::vector<std::shared_ptr<Value>> internalTopoSort();

public:
    double data;
    double grad;

    Value(double data, std::unordered_set<std::shared_ptr<Value>> children = {}, double grad = 0.0);

    std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& other);
    std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& other);
    std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& other);
    std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& other);

    std::shared_ptr<Value> pow(const double pwr); // TODO make this with other Values
    std::shared_ptr<Value> exp();

    // Activation functions
    std::shared_ptr<Value> tanh();
    std::shared_ptr<Value> relu();

    void backprop();
    friend std::ostream& operator<<(std::ostream& os, const Value& obj);
};

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b);
std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b);
std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b);
std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b);

#endif
