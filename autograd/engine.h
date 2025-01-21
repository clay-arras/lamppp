#ifndef _ENGINE_H_
#define _ENGINE_H_

#include <functional>
#include <unordered_set>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <memory>

class Value : public std::enable_shared_from_this<Value> {
private:
    std::function<void()> backward;
    std::unordered_set<std::shared_ptr<Value>> prev; 

    char op; // Debug: operation type
    std::string label; // Debug: label for the node

    std::vector<std::shared_ptr<Value>> internalTopoSort();

public:
    double data;
    double grad;

    Value(double data, std::unordered_set<std::shared_ptr<Value>> children = {}, double grad = 0.0, char op = '\0', const std::string& label = "");

    std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& other) const;
    std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& other) const;
    std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& other) const;
    std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& other) const;
    
    std::shared_ptr<Value> pow(const double pwr) const;
    std::shared_ptr<Value> exp() const;

    // Activation functions
    std::shared_ptr<Value> tanh() const;
    std::shared_ptr<Value> relu() const;

    void backprop();
    friend std::ostream& operator<<(std::ostream& os, const Value& obj);
};

#endif