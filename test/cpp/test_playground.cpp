#include "autograd/autograd_umbrella.h"

int main() {
  autograd::TensorImpl a({7, 4, 2, 4}, {2, 2});
  autograd::TensorImpl b({2, 1, -1, 3, 4, 5}, {2, 3});
  autograd::TensorImpl c = autograd::EigenBackend().matmul(a, b);
  autograd::TensorImpl d = autograd::CudaBackend().matmul(a, b);


  assert(c.data == d.data);
  assert(c.shape == d.shape);
}
