
#ifndef _ENGINE_H_
#define _ENGINE_H_

#include <functional>
#include <unordered_set>
#include <string>
#include <cmath>

class Value
{
private:
    std::function<void(void)> backward;
    std::unordered_set<std::shared_ptr<Value>> prev; // TODO: might have to make shared pointers

    char op; // debug, if non leaf node then is a combination
    string label; // debug

    void internalTopoSort(const *Value node);

public:
    double data;
    double grad;
    Value(double data, unordered_set<shared_ptr<Value>> children = {}, double grad = 0.0, char op = '\0', const std::string& label = "");

    Value operator+(const Value& other) const;
    Value operator-(const Value& other) const;
    Value operator*(const Value& other) const;
    Value operator/(const Value& other) const;
    Value pow(const double pwr) const;
    Value exp() const;

    // activation functions
    Value tanh() const;
    Value relu() const;

    void backward();
    friend std::ostream& operator<<(std::ostream& os, const Value& obj);
}

#endif