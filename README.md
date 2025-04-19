# Lamp

A lightweight C++ machine learning library from scratch, using Eigen as a backend. This project includes MNIST digit classification to demonstrate the capabilities of the autograd engine.

## Architecture

The project is organized into several key components:

- **Engine Module**: Core autograd implementation with automatic differentiation

  - `Variable` class with operator overloading for computational graph construction
  - `Function` and specialized operation classes for forward/backward propagation
  - `Tensor` class for n-dimensional array operations
  - Backward pass with topological sort for efficient gradient computation

- **Operations**: Rich set of differentiable operations

  - Basic operations: add, subtract, multiply, divide
  - Unary operations: exp, log, relu
  - Matrix operations: matrix multiplication, transpose
  - Reduction operations: sum, max

- **MNIST Example**: Digit classification implementation

  - Neural network implementation using the autograd framework
  - Data loading and preprocessing utilities

- **Utility Module**: Supporting functionality
  - CSV data loading utilities for datasets
  - Eigen integration for efficient matrix operations

## Requirements

- C++17 compatible compiler (g++-14 recommended)
- CMake 3.10+
- Eigen 3.4+ for matrix operations
- Google Benchmark (for running benchmarks)
- clang-format and clang-tidy for code formatting and static analysis

## Features

- **Modern C++ Design**: Use of templates, CRTP pattern, and smart pointers
- **Comprehensive Autograd Engine**: Support for a wide range of operations with automatic differentiation
- **Performance Optimizations**: Efficient memory management and matrix operations
- **Eigen Integration**: Seamless integration with the Eigen library

## Building and Running

```bash
# Clone the repository
git clone https://github.com/clay-arras/lamp.git
cd lamp

# Create build directory and generate build files
mkdir -p build
cd build
cmake ..

# Build the project
cmake --build .

# Run MNIST example
./mnist

# Run tests
./test_playground

# Format code
cmake --build . --target format
```

## Project Structure

```
autograd/
├── engine/              # Core autograd components
│   ├── functions/       # Implementations of differentiable operations
│   │   ├── basic_ops.*  # Addition, subtraction, multiplication, division
│   │   ├── matrix_ops.* # Matrix operations (matmul, transpose)
│   │   ├── reduct_ops.* # Reduction operations (sum, max)
│   │   └── unary_ops.*  # Unary operations (exp, log, relu)
│   ├── variable.*       # Variable class for autograd
│   ├── function.*       # Base function class
│   ├── forward_function.* # Template for forward operations
│   └── tensor.*         # Tensor implementation
├── examples/            # Example implementations
│   └── mnist.*          # MNIST digit classifier
└── util/                # Utility functions
    └── csv_reader.*     # CSV file loading utility

test/
├── cpp/                 # C++ tests
├── python/              # Python tests
└── benchmarks/          # Performance benchmarks
```

## Future Developments

- GPU acceleration
- More neural network layers
- Optimization algorithms (SGD, Adam, etc.)
- Additional operation support
