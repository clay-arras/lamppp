#include "fast_mnist.h"
#include "autograd/wrapper_engine.h"

int main() { 
    auto [data, label] = readCSV("data/mnist_dummy.csv");
    int N = (int)data.size();

    int nin = 28 * 28;
    FastLayer W1(nin, 256);
    FastLayer W2(256, 10);

    // std::function<Eigen::Matrix<SharedValue, Eigen::Dynamic, 1>(Eigen::Matrix<SharedValue, Eigen::Dynamic, 1>&)> relu = [](Eigen::Matrix<SharedValue, Eigen::Dynamic, 1>& x) {
    //     auto reluFunc = [](const SharedValue& v) -> SharedValue { 
    //         return SharedValue(v.getPtr()->relu());
    //     };
    //     x = x.unaryExpr(reluFunc);
    //     return x;
    // };
    
    // std::function<Eigen::Matrix<SharedValue, Eigen::Dynamic, 1>(Eigen::Matrix<SharedValue, Eigen::Dynamic, 1>&)> softmax = [&](Eigen::Matrix<SharedValue, Eigen::Dynamic, 1>& x) {
    //     assert(x.rows() == 10);
    //     x = x.unaryExpr([](const SharedValue& v) { return v.exp(); });
    //     SharedValue denom = SharedValue(1e-4) + x.sum();
    //     x = x / denom;
    //     return x;
    // };
    
    auto forward = [&](Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> &x) {
        // Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> Z1 = W1(x, relu);
        // Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> Z2 = W2(Z1, softmax);
        auto identity = [](Eigen::Matrix<SharedValue, Eigen::Dynamic, 1>& x) { return x; };
        Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> Z1 = W1(x, identity);
        Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> Z2 = W2(Z1, identity);
        return Z2;
    };

    std::vector<Eigen::Matrix<SharedValue, Eigen::Dynamic, 1>> y_pred;
    for (const std::vector<double>& item : data) {
        Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> x(item.size(), 1);

        std::vector<SharedValue> sharedValues;
        sharedValues.reserve(item.size());
        for (double val : item) {
            sharedValues.push_back(SharedValue(val));
        }
        x = Eigen::Map<Eigen::Matrix<SharedValue, Eigen::Dynamic, 1>>(
            sharedValues.data(), sharedValues.size());
        y_pred.push_back(forward(x));
    }
    // SharedValue loss = SharedValue(0);

    // for (int i = 0; i < N; i++) {
    //     SharedValue cross_entropy = SharedValue(0);
    //     for (int j = 0; j < 10; j++) {
    //         if (j == label[i]) {
    //             SharedValue pred_value = y_pred[i](j, 0) + SharedValue(1e-10);
    //             cross_entropy = cross_entropy + pred_value.log();
    //         }
    //     }
    //     loss = loss - cross_entropy;
    // }
    // std::cout << loss.getData() << std::endl;
}