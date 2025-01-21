#include "../autograd/nn.h"
#include "../autograd/engine.h"
#include <iostream>

int main() {
    auto net = std::make_shared<MultiLayerPerceptron>(3, std::vector<int>{4, 4, 1});

    std::vector<std::vector<double>> xs = {
        {2.0, 3.0, -1.0},
        {3.0, -1.0, 0.5},
        {0.5, 1.0, 1.0},
        {1.0, 1.0, -1.0}
    };

    std::vector<double> ys = {1.0, -1.0, -1.0, 1.0};
    std::vector<std::vector<double>> ypred;
    for (const auto& x : xs) {
        ypred.push_back((*net)(x));
    }    

    double alpha = 0.01;

    for (int i = 0; i < 100; i++) {
        std::vector<std::vector<double>> ypred;
        for (const auto& x : xs) {
            ypred.push_back((*net)(x));
        }
        for (auto i : ypred) {
            for (auto j : i)
                std::cout << j << " ";
        }
        std::cout << std::endl;

        auto loss = std::make_shared<Value>(0.0);
        for (size_t j = 0; j < ys.size(); j++) {
            auto diff = std::make_shared<Value>(ypred[j][0]) - std::make_shared<Value>(ys[j]);
            loss = loss + (diff->pow(2.0));
        }
        std::cout << loss->data << std::endl;

        loss->backprop();
        auto params = net->parameters();
        for (auto& param : params) {
            param->data -= alpha * param->grad;
            param->grad = 0.0;
        }
    }

    return 0;
}