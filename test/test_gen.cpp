#include <Eigen/Core>
#include "autograd/engine.h"
#include "autograd/wrapper_engine.h"
#include <functional>
#include <memory>


int main() {
    // Create a 2x2 matrix of shared_ptr<Value>
    // Eigen::Matrix<std::shared_ptr<Value>, 2, 2> mat;
    // auto fn = [](const std::shared_ptr<Value> &a) { 
    //     return std::make_shared<Value>(0.0f); 
    // };
    // mat = mat.unaryExpr(fn);

    // // Print the matrix values
    // std::cout << "Matrix:" << std::endl;
    // for (int i = 0; i < mat.rows(); ++i) {
    //     for (int j = 0; j < mat.cols(); ++j) {
    //         std::cout << mat(i, j)->data << " ";
    //     }
    //     std::cout << std::endl;
    // }
    
    // Eigen::Matrix<std::shared_ptr<Value>, 2, 2> result = mat.unaryExpr(std::mem_fn(&Value::exp)); 
    // std::cout << "Exponent Matrix:" << std::endl;
    // for (int i = 0; i < result.rows(); ++i) {
    //     for (int j = 0; j < result.cols(); ++j) {
    //         std::cout << result(i, j)->data << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // ---
    Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic> mat1(784, 256);
    Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic> mat2(256, 10);

    auto init_fn = [](const SharedValue&) { 
        return SharedValue((2.0f * (static_cast<float>(rand()) / RAND_MAX) - 1.0f) / 1000);
    };

    mat1 = mat1.unaryExpr([&init_fn](const SharedValue& val) { return init_fn(val); });
    mat2 = mat2.unaryExpr([&init_fn](const SharedValue& val) { return init_fn(val); });

    Eigen::Matrix<SharedValue, Eigen::Dynamic, Eigen::Dynamic> res = mat1 * mat2;

    // Eigen::MatrixXf mat1(784, 256);
    // Eigen::MatrixXf mat2(256, 10);

    // auto init_fn = [](float) { 
    //     return (2.0f * (static_cast<float>(rand()) / RAND_MAX) - 1.0f) / 1000;
    // };

    // mat1 = mat1.unaryExpr([&init_fn](float) { return init_fn(0); });
    // mat2 = mat2.unaryExpr([&init_fn](float) { return init_fn(0); });
    // Eigen::MatrixXf res = mat1 * mat2;


    return 0;
}
