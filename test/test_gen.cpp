#include <Eigen/Core>
#include "autograd/engine.h"
#include <functional>


int main() {
    // Create a 2x2 matrix of shared_ptr<Value>
    Eigen::Matrix<std::shared_ptr<Value>, 2, 2> mat;
    mat(0, 0) = std::make_shared<Value>(1.0f);
    mat(0, 1) = std::make_shared<Value>(2.0f);
    mat(1, 0) = std::make_shared<Value>(3.0f);
    mat(1, 1) = std::make_shared<Value>(4.0f);
    
    // Print the matrix values
    std::cout << "Matrix:" << std::endl;
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            std::cout << mat(i, j)->data << " ";
        }
        std::cout << std::endl;
    }
    
    Eigen::Matrix<std::shared_ptr<Value>, 2, 2> result = mat.unaryExpr(std::mem_fn(&Value::exp)); 
    std::cout << "Exponent Matrix:" << std::endl;
    for (int i = 0; i < result.rows(); ++i) {
        for (int j = 0; j < result.cols(); ++j) {
            std::cout << result(i, j)->data << " ";
        }
        std::cout << std::endl;
    }
    // std::shared_ptr<Value> value_ptr = mat(1, 0);
    // std::shared_ptr<Value> result = std::make_shared<Value>(std::sqrt(value_ptr->data));
    // std::cout << "Square root of " << value_ptr->data << " is " << result->data << std::endl;
    
    return 0;
}
