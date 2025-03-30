#include "fast_mnist.h"
#include "autograd/wrapper_engine.h"

int main() { 
    auto [data, label] = readCSV("data/mnist_dummy.csv");
    int N = (int)data.size();

    int nin = 28 * 28;
    FastLayer W1(nin, 256);
    FastLayer W2(256, 10);

    std::function<Eigen::Matrix<SharedValue, Eigen::Dynamic, 1>(Eigen::Matrix<SharedValue, Eigen::Dynamic, 1>)> relu = [](Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> x) {
        auto reluFunc = [](const SharedValue& v) -> SharedValue { 
            return SharedValue(v.getPtr()->relu());
        };
        Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> z;
        z = x.unaryExpr(reluFunc);
        return z;
    };
    
    std::function<Eigen::Matrix<SharedValue, Eigen::Dynamic, 1>(Eigen::Matrix<SharedValue, Eigen::Dynamic, 1>)> softmax = [&](Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> x) {
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
    
    auto forward = [&](Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> x) {
        Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> Z1 = W1(x, relu);
        Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> Z2 = W2(Z1, softmax);
        return Z2;
    };


    std::vector<Eigen::Matrix<SharedValue, Eigen::Dynamic, 1>> y_pred;
    for (const std::vector<double>& item : data) {
        Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> x;
        x.resize((int)item.size(), 1);
        for (size_t i = 0; i < item.size(); ++i) {
            x(i, 0) = SharedValue(item[i]);
        }
        y_pred.push_back(forward(x));
    }
    SharedValue loss = SharedValue(0);

    std::cout << "Predictions (y_pred) in a grid format:" << std::endl;
    for (const auto& prediction : y_pred) {
        for (int i = 0; i < prediction.rows(); ++i) {
            std::cout << prediction(i, 0).getData() << " ";
        }
        std::cout << std::endl;
    }

    for (int i = 0; i < N; i++) {
        SharedValue cross_entropy = SharedValue(0);
        for (int j = 0; j < 10; j++) {
            if (j == label[i]) {
                SharedValue pred_value = y_pred[i](j, 0) + SharedValue(1e-10);
                cross_entropy = cross_entropy + pred_value.log();
            }
        }
        loss = loss - cross_entropy;
    }
    loss = loss / SharedValue(N);
    std::cout << "Loss: " << loss.getData() << std::endl;
}