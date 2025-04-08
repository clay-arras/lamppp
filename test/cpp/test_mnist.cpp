#include "autograd/engine/wrapper_engine.h"
#include "autograd/nn/fast_layer.h"
#include "autograd/util/csv_reader.h"

namespace {

SharedValue relu_func(const SharedValue& v) {
    return SharedValue(v.getPtr()->relu());
}

SharedValue exp_func(const SharedValue& v) {
    return v.exp();
}

Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic> relu(Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic>& x) {
    x = x.unaryExpr(&relu_func);
    return x;
}

Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic> softmax(Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic>& x) {
    // for (auto row=x.rowwise().begin(); row != x.rowwise().end(); ++row) {
    //   *row = row->unaryExpr(&exp_func);
    //   SharedValue denom = SharedValue(1e-4) + row->sum();
    //   *row = *row / denom;
    // }
    return x;
}

Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic> forward(
    Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic>& x, FastLayer& w1,
    FastLayer& w2) {
    Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic> z1 = w1(x, relu);
    Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic> z2 = w2(z1, softmax);
    return z2;
}

}

int main() {
    auto [data, label] = readCSV("data/mnist_dummy.csv");
    data.resize(1000);
    label.resize(1000);
    int n = static_cast<int>(data.size());

    int nin = 28 * 28;
    FastLayer w1(nin, 256);
    FastLayer w2(256, 10);

    Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic> input_matrix(data.size(), 28 * 28);
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[i].size(); ++j) {
            input_matrix(i, j) = SharedValue(data[i][j]);
        }
    }

    Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic> y_pred = forward(input_matrix, w1, w2);
    return 0;
}