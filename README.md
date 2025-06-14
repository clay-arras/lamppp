<div align="left">
  <img src="https://github.com/user-attachments/assets/52f467bf-bc40-4e01-8389-358d74777731" alt="neural_bulb_svg (3)" width="600">
</div>

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/clay-arras/lamp) <!-- Placeholder -->
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/clay-arras/lamp/blob/main/LICENSE) <!-- Placeholder -->
[![Version](https://img.shields.io/badge/version-0.1.0-blue)](https://github.com/clay-arras/lamp) <!-- Placeholder -->

# Lamp++

**A C++ automatic differentiation engine and tensor library built to capture 80% of Pytorch's functionality with less than 2% of the code.**

Have you ever looked into PyTorch's codebase and been overwhelmed by complex macros, codegen scripts, and layers of abstraction? Lamp++ is the opposite: a clean, concise implementation of automatic differentiation and n-dim tensors.

## What is Lamp++?

Lamp++ is a from-scratch implementation of the core concepts behind modern deep learning frameworks, written in about 4,000 lines of C++ and CUDA. It's built around a simple philosophy: **"What I cannot create, I do not understand"** (Richard Feynman).

**What makes it different:**
- **Small and readable**: ~1,500 lines for autograd, ~2,500 lines for tensors. Zero dependencies 
- **Transparent**: No mysterious compilation steps or hidden optimizations
- **Hackable**: Want to add a new operation? It's straightforward
- **Fast**: Lamp++ is about 3x faster than Pytorch on CUDA operation benchmarks. See [benchmarks](https://github.com/clay-arras/lamppp/blob/main/benchmarks/README.md).

**What it's good for:**
- Learning how automatic differentiation actually works
- Understanding the fundamentals behind PyTorch/TensorFlow
- Building small and fast neural networks

**What it's not:**
- A production-ready framework to build complex neural architectures (maybe one day ;])
- Optimized for massive models or distributed training

## Quick Start

### Building

```bash
git clone https://github.com/clay-arras/lamp.git
cd lamp
cmake -S . -B build -DENABLE_CUDA=ON  # or OFF if you don't have CUDA
cmake --build build
```

### Your first neural network

```cpp
#include "lamppp/lamppp.hpp"

int main() {
    // Create some tensors
    lmp::Tensor data_a(std::vector<float>{2.0f, 4.0f}, {1, 2}, 
                       lmp::DeviceType::CUDA, lmp::DataType::Float32);
    lmp::Tensor data_b(std::vector<float>{3.0f, 1.0f}, {1, 2}, 
                       lmp::DeviceType::CUDA, lmp::DataType::Float32);

    // Wrap them in Variables to track gradients
    lmp::Variable a(data_a, true);
    lmp::Variable b(data_b, true);

    // Do some math
    lmp::Variable c = a * b;
    lmp::Variable loss = lmp::sum(lmp::sum(c, 1), 0);

    // Compute gradients automatically (wow :o)
    loss.backward(); 

    std::cout << "Gradient of a: " << a.grad() << std::endl;
    std::cout << "Gradient of b: " << b.grad() << std::endl;
}
```

Want to see a complete neural network? Check out `examples/mnist.cpp` for a full implementation that trains on MNIST.

## How it works

Lamp++ builds computation graphs dynamically as you run your code. When you do `a * b`, it creates a multiplication node that remembers how to compute gradients. When you call `loss.backward()`, it walks backward through the graph computing derivatives using the chain rule.

The tensor library stores everything as `void*` with type information, so you can add new data types easily. CUDA operations have their own kernels in `src/tensor/cuda/`, while CPU operations live in `src/tensor/native/`.

If you want to add a new operation, inherit from `autograd::Function` and implement `forward` and `backward` methods. 

## Requirements

- **Required**: CMake 3.24+, C++20 compiler, OpenMP
- **Optional**: CUDA toolkit (for GPU acceleration -- although keep in mind that most of the optimization was done for CUDA, not CPU, so it's highly recommended)
- **For tests**: Python 3.11+ (stress testing against Pytorch)


