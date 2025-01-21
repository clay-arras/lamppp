#ifndef _NN_H_
#define _NN_H_

#include <vector>
#include <memory>
#include <random>
#include "engine.h"

class Neuron : public std::enable_shared_from_this<Neuron> {
private:
    std::vector<std::shared_ptr<Value>> weights;
    std::shared_ptr<Value> bias;

public:
    Neuron(int nin);
    std::vector<std::shared_ptr<Value>> parameters();
    double operator()(std::vector<double> x);
};


class Layer : public std::enable_shared_from_this<Layer> {
private:
    std::vector<std::shared_ptr<Neuron>> neurons;

public:
    Layer(int nin, int nout);
    std::vector<std::shared_ptr<Value>> parameters();
    std::vector<double> operator()(std::vector<double> x);
};


class MultiLayerPerceptron : public std::enable_shared_from_this<MultiLayerPerceptron> {
private:
    std::vector<std::shared_ptr<Layer>> layers;

public:
    MultiLayerPerceptron(int nin, std::vector<int> nouts);
    std::vector<std::shared_ptr<Value>> parameters();
    std::vector<double> operator()(std::vector<double> x);
};

#endif _NN_H_