#include <iostream>
#include "autograd/engine/backend/cuda_backend.h"
#include "autograd/engine/tensor_impl.h"

int main() {

  autograd::TensorImpl a({1, 2, 3, 4, 5, 6}, {6});
  autograd::TensorImpl b({4, 2, 6, 1, 3, 7}, {6});

  autograd::TensorImpl c = autograd::CudaBackend().add(a, b);
  for (float i : c.data) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
}
