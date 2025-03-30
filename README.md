# MicroGrad C++

A from-scratch implementation of autograd in both Python and C++, inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd), with MNIST digit classification in multiple implementations.

## Core Implementations

- **Pure Python Autograd Engine**: Complete autograd implementation from scratch in Python
- **Pure C++ Autograd Engine**: Full autograd system rebuilt from the ground up in C++
- **MNIST Implementations**:
  - Using only basic PyTorch vector operations
  - Using only our custom Python autograd engine
  - Using our C++ autograd engine from scratch

## Architecture

The project is organized into several key components:

- **Engine Module**: Core autograd implementation with automatic differentiation

  - `Value` class with operator overloading for computational graph construction
  - Backward pass with topological sort for efficient gradient computation
  - Support for higher-order derivatives

- **Neural Network Module**: Modular neural network building blocks

  - `Neuron`, `Layer`, and `MultiLayerPerceptron` classes for network construction
  - Support for activation functions (ReLU, tanh) and customizable architectures
  - Fast layer implementations for optimized performance

- **MNIST Module**: Implementations for digit classification

  - Standard implementation with automatic differentiation
  - Fast implementation with optimized batch processing

- **Utility Module**: Supporting functionality
  - CSV data loading utilities for datasets
  - Eigen integration for matrix operations with autograd values

## Requirements

- C++17 compatible compiler (g++-14 recommended)
- CMake 3.10+
- Python 3.11+
- Eigen 3.4+ (for matrix operations)
- clang-format (for code formatting)
- clang-tidy (for static analysis)

## Features

- **Comprehensive Autograd Engine**: Support for operations (+, -, \*, /, pow, exp, log) with automatic differentiation
- **Modern C++ Design**: Extensive use of smart pointers and RAII principles
- **Neural Network Framework**: Modular architecture with customizable layers and activation functions
- **Performance Optimizations**: Batch processing capabilities with mathematical optimizations
- **Eigen Integration**: Seamless integration with the Eigen library for efficient matrix operations

## Building and Running

```bash
# Clone the repository
git clone https://github.com/yourusername/autograd_cpp.git
cd autograd_cpp

# Create build directory and generate build files
mkdir -p build
cd build
cmake ..

# Build the project
cmake --build .

# Run MNIST example
./mnist

# Run tests
./test_engine
./test_nn
./test_mnist
./test_fast_mnist

# Format code (optional)
cmake --build . --target format
```

## Future Developments

- Improved parallelization strategies
- CUDA support for GPU acceleration
- Expanded operation support
- Optimization algorithms (Adam, RMSProp)
