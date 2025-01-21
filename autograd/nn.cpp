#include "nn.h"

Neuron::Neuron(int nin) {
    std::random_device seed;
    std::mt19937 gen{seed()};
    std::uniform_real_distribution<> dist{-1.0, 1.0};
    for (int i=0; i<nin; i++) {
        this->weights.push_back(std::make_shared<Value>(dist(gen)));
    }
    this->bias = std::make_shared<Value>(dist(gen));
}

std::vector<std::shared_ptr<Value>> Neuron::parameters() {
    std::vector<std::shared_ptr<Value>> ret(this->weights.begin(), this->weights.end());
    ret.push_back(this->bias);
    return ret;
}

// TODO: look into parallized implementations
double Neuron::operator()(std::vector<double> x) {
    double ret = this->bias->data;
    for (int i=0; i<(int)x.size(); i++) {
        ret += (this->weights[i] * std::make_shared<Value>(x[i]))->data;
    }
    return ret;
}

Layer::Layer(int nin, int nout) {
    for (int i=0; i<nout; i++) {
        this->neurons.push_back(std::make_shared<Neuron>(nin));
    }
}

std::vector<std::shared_ptr<Value>> Layer::parameters() {
    std::vector<std::shared_ptr<Value>> params;
    for (int i=0; i<(int)this->neurons.size(); i++) {
        std::vector<std::shared_ptr<Value>> n_params = this->neurons[i]->parameters();
        params.insert(params.end(), n_params.begin(), n_params.end());
    }
    return params;
}

std::vector<double> Layer::operator()(std::vector<double> x) {
    std::vector<double> ret;
    for (int i=0; i<(int)this->neurons.size(); i++) {
        ret.push_back((*this->neurons[i])(x));
    }
    return ret;
}

MultiLayerPerceptron::MultiLayerPerceptron(int nin, std::vector<int> nouts) {
    int prev = nin;
    for (int i=0; i<(int)nouts.size(); i++) {
        this->layers.push_back(std::make_shared<Layer>(prev, nouts[i]));
        prev = nouts[i];
    }
}

std::vector<std::shared_ptr<Value>> MultiLayerPerceptron::parameters() {
    std::vector<std::shared_ptr<Value>> params;
    for (int i=0; i<(int)this->layers.size(); i++) {
        std::vector<std::shared_ptr<Value>> l_params = this->layers[i]->parameters();
        params.insert(params.end(), l_params.begin(), l_params.end());
    }
    return params;
}

std::vector<double> MultiLayerPerceptron::operator()(std::vector<double> x) {
    std::vector<double> ret = x;
    for (int i=0; i<(int)this->layers.size(); i++) {
        ret = (*this->layers[i])(ret);
    }
    return ret;
}