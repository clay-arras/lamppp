# MicroGrad C++

A C++ implementation of autograd inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) - a tiny autograd engine that implements backpropagation (reverse-mode autodiff) over a dynamically built DAG.

## Overview

This project implements:

1. An autograd engine that tracks computations and enables automatic differentiation
2. A neural network library built on top of the autograd engine
3. Support for basic neural network operations including:
   - Forward and backward propagation
   - Common activation functions (tanh, ReLU)
   - Multi-layer perceptron architecture

## Building

Requirements:

- C++17 compatible compiler (g++-14 recommended)
- Make

## Features

- **Autograd Engine**

  - Automatic differentiation through computational graphs
  - Support for basic mathematical operations (+, -, \*, /, pow)
  - Activation functions (tanh, ReLU)

- **Neural Network Library**
  - Modular design with Neuron, Layer, and MLP classes
  - Support for multi-layer architectures
  - Automatic parameter management

## License

MIT License

## Acknowledgments

This project is inspired by Andrej Karpathy's micrograd.
