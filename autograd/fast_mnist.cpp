#include "fast_mnist.h"

int main() { 
    auto [data, label] = readCSV("data/mnist_dummy.csv");
    int N = (int)data.size();

    int nin = 28 * 28;
    FastLayer W1(nin, 256);
    FastLayer W2(256, 10);

    auto softmax = [&](Eigen::Matrix<std::shared_ptr<Value>, Eigen::Dynamic, 1> x) -> Eigen::Matrix<std::shared_ptr<Value>, Eigen::Dynamic, 1> {
        assert(x.rows() == 10);
        x = x.unaryExpr(&Value::exp);
        std::shared_ptr<Value> denom = std::make_shared<Value>(1e-4) + x.sum();
        x = x / denom;
        return x;
    };

    Eigen::Matrix<std::shared_ptr<Value>, Eigen::Dynamic, 1> test_input(10, 1);
    for (int i = 0; i < 10; ++i) {
        test_input(i, 0) = std::make_shared<Value>((float)i); // Test input values from 0 to 9
    }
    Eigen::Matrix<std::shared_ptr<Value>, Eigen::Dynamic, 1> softmax_output = softmax(test_input);

    std::cout << "Softmax Output:" << std::endl;
    for (int i = 0; i < softmax_output.rows(); ++i) {
        std::cout << softmax_output(i, 0)->data << " ";
    }
    std::cout << std::endl;
    
    // auto forward = [&](std::vector<std::shared_ptr<Value>> x) {
    //     std::vector<std::shared_ptr<Value>> Z1 = W1(x);
    //     std::vector<std::shared_ptr<Value>> Z2 = W2(Z1, false);
    //     return softmax(Z2);
    // };

    // std::vector<std::vector<std::shared_ptr<Value>>> y_pred;
    // for (std::vector<double> item : data) {
    //     std::vector<std::shared_ptr<Value>> ptrs;
    //     for (double i : item)
    //         ptrs.push_back(std::make_shared<Value>(Value(i)));
    //     y_pred.push_back(forward(ptrs));
    // }
    // std::shared_ptr<Value> loss = std::make_shared<Value>(Value(0));

    // for (int i=0; i<N; i++) {
    //     std::shared_ptr<Value> cross_entropy = std::make_shared<Value>(Value(0));
    //     for (int j=0; j<10; j++) 
    //         if (j == label[i])
    //             cross_entropy = cross_entropy + y_pred[i][j]->log();
    //     loss = loss - cross_entropy;
    // }
    // std::cout << loss->data << std::endl;
}