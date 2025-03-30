#include "fast_mnist.h"

int main() { 
    auto [data, label] = readCSV("data/mnist_dummy.csv");
    int N = (int)data.size();

    int nin = 28 * 28;
    FastLayer W1(nin, 256);
    FastLayer W2(256, 10);

    auto softmax = [&](Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> x) -> Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> {
        assert(x.rows() == 10);
        x = x.unaryExpr([](const SharedValue& v) { return v.exp(); });
        SharedValue denom = SharedValue(1e-4) + x.sum();
        x = x / denom;
        return x;
    };

    Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> test_input;
    test_input.resize(10, 1);
    for (int i = 0; i < 10; ++i) {
        test_input(i, 0) = SharedValue(static_cast<float>(i)); // Test input values from 0 to 9
    }
    Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> softmax_output = softmax(test_input);

    std::cout << "Softmax Output:" << std::endl;
    for (int i = 0; i < softmax_output.rows(); ++i) {
        std::cout << softmax_output(i, 0).getData() << " ";
    }
    std::cout << std::endl;
    
    // auto forward = [&](std::vector<SharedValue> x) {
    //     std::vector<SharedValue> Z1 = W1(x);
    //     std::vector<SharedValue> Z2 = W2(Z1, false);
    //     return softmax(Z2);
    // };

    // std::vector<std::vector<SharedValue>> y_pred;
    // for (std::vector<double> item : data) {
    //     std::vector<SharedValue> ptrs;
    //     for (double i : item)
    //         ptrs.push_back(SharedValue(Value(i)));
    //     y_pred.push_back(forward(ptrs));
    // }
    // SharedValue loss = SharedValue(Value(0));

    // for (int i=0; i<N; i++) {
    //     SharedValue cross_entropy = SharedValue(Value(0));
    //     for (int j=0; j<10; j++) 
    //         if (j == label[i])
    //             cross_entropy = cross_entropy + y_pred[i][j].log();
    //     loss = loss - cross_entropy;
    // }
    // std::cout << loss.getData() << std::endl;
}