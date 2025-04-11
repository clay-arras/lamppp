#pragma once
#ifndef _TENSOR_H_
#define _TENSOR_H

#include <vector>
#include <Eigen/Core>

class Tensor {
public:
    std::vector<float> data;
    std::vector<int> shape;

    Tensor() = default;
    Tensor(const std::vector<float>& data, const std::vector<int>& shape)
        : data(data), shape(shape) {}

    Tensor operator+(const Tensor& other) const; 
    Tensor operator-(const Tensor& other) const; 
    Tensor operator*(const Tensor& other) const; 
    Tensor operator/(const Tensor& other) const; 

    Tensor operator+(float other) const; 
    Tensor operator-(float other) const; 
    Tensor operator*(float other) const; 
    Tensor operator/(float other) const; 

    Tensor matmul(const Tensor& other) const;
    Tensor log() const;
    Tensor exp() const;
    Tensor relu() const;

private:
    Eigen::Map<Eigen::MatrixXf> as_matrix(int rows, int cols) const {
        return Eigen::Map<Eigen::MatrixXf>(const_cast<float*>(data.data()), rows, cols);
    }
    Eigen::Map<Eigen::ArrayXf> as_array() const {
        return Eigen::Map<Eigen::ArrayXf>(const_cast<float*>(data.data()), data.size());
    }
};

#endif