**jacobian basics**
jacobian is just a local derivative in chain rule, but for vectors

- e.g. df/dx

jacobian vector products

jacobian chain rule, tensor chain rule

jacobian for higher dimensional matrices: to calculate the jacobian, you flatten the matrix of both operations into a 1d matrix, then you have a N\*M jacobian

**how to operate on high-dimensional tensors with 2d matrices backend**

- batch processing
- flattening and reshaping (into and out of 2D matrices; with Eigen::Map it's O(1))

resource: https://cs231n.stanford.edu/handouts/derivatives.pdf

matrix operations: matmul, transpose
reduction operations: sum, mean, max, min
creation operatioons: zeros, ones, rand, tensor
element-wise operations: +, -, \*, /
unary operations: relu, tanh

add some debug stuff (like cout, refactor some code)

| Category                   | Representative methods                                                   | What they expose / guarantee                                                                                                             |
| -------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **Metadata access**        | `sizes()`, `dim()`, `stride(int)`, `numel()`                             | Shape/stride information is needed by _every_ kernel and autograd node. ([PyTorch][1])                                                   |
| **Type & device**          | `dtype() / scalar_type()`, `device()`, `is_cuda()`, `layout()`           | Lets the dispatcher pick the correct backend (CPU, CUDA, Sparse, …) and code‑gen route. ([PyTorch][2])                                   |
| **Data pointer**           | `data_ptr<T>()`, `unsafeGetTensorImpl()`                                 | Gives kernels raw access to memory; kept out of TorchScript for safety. ([PyTorch][3], [PyTorch Forums][4])                              |
| **Simple state mutators**  | `resize_()`, `set_(storage, sizes, strides, offset)`, `storage_offset()` | Change view of the same storage without reallocating. Used by view/reshape/slice ops built outside the class. ([GitHub][5], [GitHub][6]) |
| **Contiguity helpers**     | `is_contiguous()`, `contiguous(memory_format)`                           | Needed so kernels know when they can assume unit‑stride memory. ([GitHub][7])                                                            |
| **Device / type casting**  | `to(options, non_blocking, copy)`, `cpu()`, `cuda()`                     | One canonical move‑once API; higher‑level wrappers (`tensor.to("cuda:0")` in Python) just call this. ([GitHub][7])                       |
| **Autograd hooks & flags** | `requires_grad()`, `set_requires_grad()`, `grad()`, `register_hook()`    | Kept in the core so every tensor—regardless of backend—can participate in autograd. ([GitHub][6])                                        |
| **Printing / debug**       | `__str__` stream overload, `toString()`                                  | Purely for debugging, but part of the base so it never relies on optional ops. ([PyTorch Forums][8])                                     |

[1]: https://pytorch.org/docs/stable/generated/torch.Tensor.size.html?utm_source=chatgpt.com "torch.Tensor.size — PyTorch 2.7 documentation"
[2]: https://pytorch.org/docs/stable/tensors.html?utm_source=chatgpt.com "torch.Tensor — PyTorch 2.7 documentation"
[3]: https://pytorch.org/docs/stable/generated/torch.Tensor.data_ptr.html?utm_source=chatgpt.com "torch.Tensor.data_ptr — PyTorch 2.7 documentation"
[4]: https://discuss.pytorch.org/t/tensor-data-ptr-not-visible-from-torchscript/216655?utm_source=chatgpt.com "'Tensor.data_ptr ()' not visible from TorchScript - PyTorch Forums"
[5]: https://github.com/pytorch/pytorch/blob/main/c10/core/TensorImpl.cpp?utm_source=chatgpt.com "pytorch/c10/core/TensorImpl.cpp at main - GitHub"
[6]: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/core/Tensor.h?utm_source=chatgpt.com "pytorch/aten/src/ATen/core/Tensor.h at main - GitHub"
[7]: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/core/Tensor.cpp?utm_source=chatgpt.com "pytorch/aten/src/ATen/core/Tensor.cpp at main - GitHub"
[8]: https://discuss.pytorch.org/t/how-to-print-at-tensor-in-c/66214?utm_source=chatgpt.com "How to print at::tensor in C++ - C++ - PyTorch Forums"
