# MicroGrad C++

A from-scratch implementation of autograd in both Python and C++, inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd), with MNIST digit classification in multiple implementations.

## Core Implementations

- **Pure Python Autograd Engine**: Complete autograd implementation from scratch in Python
- **Pure C++ Autograd Engine**: Full autograd system rebuilt from the ground up in C++
- **MNIST Implementations**:
  - Using only basic PyTorch vector operations
  - Using only our custom Python autograd engine
  - Using our C++ autograd engine from scratch

## Requirements

- C++17 compatible compiler (g++-14 recommended)
- Make
- Python 3.11+

## Features

- **Autograd Engines**: Support for operations (+, -, \*, /, pow, exp, log) with automatic differentiation
- **Neural Network**: Modular architecture with customizable layers and activation functions
- **Performance**: Batch processing capabilities with optimization experiments

## Future Developments

- Improved parallelization strategies
- Vector operations optimization
- CUDA support

## License

MIT License

This project is inspired by Andrej Karpathy's micrograd.
