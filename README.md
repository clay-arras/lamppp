# Lamp++

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/clay-arras/lamp) <!-- Placeholder -->
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/clay-arras/lamp/blob/main/LICENSE) <!-- Placeholder -->
[![Version](https://img.shields.io/badge/version-0.1.0-blue)](https://github.com/clay-arras/lamp) <!-- Placeholder -->

Lamp++ is a C++ automatic differentiation (autograd) engine and tensor library, built from scratch with a focus on performance, extensibility, and modern C++ design. It aims to provide a transparent and adaptable foundation for machine learning research and development.

Performance is a key focus, with benchmarks indicating speeds comparable to PyTorch for many tensor operations. The library supports CUDA for GPU acceleration and leverages Eigen for optimized CPU matrix math.

## Core Design & Capabilities

- **Dynamic Autograd Engine:** Lamp++ builds computation graphs dynamically. The core `autograd::Variable` class wraps `tensor::Tensor` objects, tracking operations to enable automatic gradient calculation via a backward pass that uses a topological sort. New operations can be added by inheriting from the `autograd::Function` base class and implementing static `forward` and `backward` methods.

- **Type-Erased and Extensible Tensor Library:**

  - The `tensor::Tensor` stores data using `void*` and a `tensor::DataType` enum (supporting types like `float`, `double`, `int` etc.). This type erasure is managed through the `LMP_DISPATCH_ALL_TYPES` macro, which allows for new data types to be added by extending the enum and the dispatch logic.
  - Supports CPU and CUDA devices, with explicit data management.
  - Tensor operations like `reshape`, `expand_dims`, and broadcasting (via an internal `AlignUtil` that computes aligned shapes and strides for element-wise operations) are available.

- **CUDA Acceleration:** Many tensor operations have custom CUDA kernels (found in `src/tensor/cuda/`) for GPU performance. Device selection is explicit, offering control over data locality.

- **Modern C++:** Developed using C++17, employing templates, CRTP for static polymorphism (e.g., in operation dispatch), and smart pointers for memory safety.

## Getting Started

### Requirements

- C++17 compatible compiler (e.g., GCC 9+, Clang 10+)
- CMake (3.10+ recommended)
- Eigen (3.4+ recommended)
- NVIDIA CUDA Toolkit (optional, for GPU support, 11.x+ recommended)
- Google Benchmark (optional, for running benchmarks)

### Building Lamp++

```bash
git clone https://github.com/clay-arras/lamp.git # Or your repository URL
cd lamp
mkdir -p build && cd build
cmake .. # Add -DLAMP_ENABLE_CUDA=ON to enable CUDA
cmake --build . --config Release
```

## Usage Example

A minimal example demonstrating autograd:

```cpp
#include <iostream>
#include "lamppp/lamppp.hpp"

using lmp::autograd::Variable;
using lmp::tensor::Tensor;
using lmp::tensor::DeviceType;
using lmp::tensor::DataType;
using namespace lmp::autograd::ops;

int main() {

    Tensor data_a(std::vector<float>{2.0f, 4.0f}, {1, 2}, DeviceType::CUDA, DataType::Float32);
    Tensor data_b(std::vector<float>{3.0f, 1.0f}, {1, 2}, DeviceType::CUDA, DataType::Float32);

    Variable a(data_a, true);
    Variable b(data_b, true);

    Variable c = a * b;
    Variable loss = sum(sum(c, 1), 0);

    loss.backward();

    std::cout << "Gradient of a (should be data of b: {3.0, 1.0}):\n" << a.grad() << std::endl;
    std::cout << "Gradient of b (should be data of a: {2.0, 4.0}):\n" << b.grad() << std::endl;
}
```

_(Note: Tensor initialization and operation naming have been verified. The `using namespace lmp::autograd::ops;` brings `mul` and `sum` into scope. The reduction to scalar loss is done by summing over each axis sequentially.)_

## Project Structure Highlights

- **`include/lamppp/`**: Public API headers.
  - `autograd/`: Components like `Variable`, `Function`.
  - `tensor/`: `Tensor`, `DataType`, `DeviceType`, operation declarations.
- **`src/`**: Implementation details.
  - `autograd/`: Core autograd logic.
  - `tensor/`: Tensor operations, including `cuda/` for GPU kernels and `native/` for CPU/dispatch.
- **`examples/`**: Contains usage examples like the MNIST digit classifier.
- **`benchmarks/`**: Performance tests.
- **`test/`**: Unit and integration tests.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and open a pull request. Focus areas include new operations, optimizer algorithms, or enhanced GPU kernel coverage.

## Future Considerations

- Expanding the set of neural network layers and optimizers.
- Exploring further graph optimizations.
- Broadening hardware backend support.

## License

Distributed under the MIT License. See the `LICENSE` file for details.
