#include "mnist.h"

int main() { 
    auto [data, label] = readCSV("data/mnist_dummy.csv");
    int N = (int)data.size();

    int nin = 28 * 28;
    Layer W1(nin, 256);
    Layer W2(256, 10);

    auto softmax = [&](std::vector<std::shared_ptr<Value>> x) -> std::vector<std::shared_ptr<Value>> {
        assert((int)x.size() == 10);
        std::shared_ptr<Value> denom = std::make_shared<Value>(Value(1e-4));
        for (auto i : x)
            denom = denom + i->exp();
        for (auto &i : x)
            i = i->exp() / denom;
        return x;
    };

    auto forward = [&](std::vector<std::shared_ptr<Value>> x) {
        std::vector<std::shared_ptr<Value>> Z1 = W1(x);
        std::vector<std::shared_ptr<Value>> Z2 = W2(Z1, false);
        return softmax(Z2);
    };

    std::vector<std::vector<std::shared_ptr<Value>>> y_pred;
    for (std::vector<double> item : data) {
        std::vector<std::shared_ptr<Value>> ptrs;
        for (double i : item)
            ptrs.push_back(std::make_shared<Value>(Value(i)));
        y_pred.push_back(forward(ptrs));
    }
    std::shared_ptr<Value> loss = std::make_shared<Value>(Value(0));

    for (int i=0; i<N; i++) {
        std::shared_ptr<Value> cross_entropy = std::make_shared<Value>(Value(0));
        for (int j=0; j<10; j++) 
            if (j == label[i])
                cross_entropy = cross_entropy + y_pred[i][j]->log();
        loss = loss - cross_entropy;
    }
    std::cout << loss->data << std::endl;
}