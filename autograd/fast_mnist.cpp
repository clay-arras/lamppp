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

    auto forward = [&](Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> x) {
        Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> Z1 = W1(x, relu);
        Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> Z2 = W2(Z1, softmax);
        return Z2;
    };

    Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic> X_batch(nin, N);
    for (int i = 0; i < N; ++i) {
        for (size_t j = 0; j < data[i].size(); ++j) {
            X_batch(j, i) = SharedValue(data[i][j]);
        }
    }

    Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic> predictions(10, N);

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        Eigen::Matrix<SharedValue, Eigen::Dynamic, 1> pred = forward(X_batch.col(i));
        predictions.col(i) = pred;
    }
    
    SharedValue loss = SharedValue(0);
    
    for (int i = 0; i < N; ++i) {
        int true_label = label[i];
        SharedValue pred_value = predictions(true_label, i) + SharedValue(1e-10);
        loss = loss - pred_value.log();
    }
    
    loss = loss / SharedValue(N);
    std::cout << "Loss: " << loss.getData() << std::endl;
}