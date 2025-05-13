import lamppp
import torch
from sample_helper import sample_matrices


class Operation:
    def __init__(self):
        self.backward_atol_mult = 2
        self.rtol = 1e-5
        self.atol = None
        self.sampler = None
        self.ranges = None
        self.torch_fn = None
        self.cpp_fn = None


class Add(Operation):
    def __init__(self):
        super().__init__()
        self.atol = 1e-6
        self.ranges = [[-1000, 1000]]
        self.sampler = lambda: sample_matrices(2, self.ranges)
        self.torch_fn = torch.add
        self.cpp_fn = lamppp.add


class Sub(Operation):
    def __init__(self):
        super().__init__()
        self.atol = 1e-6
        self.sampler = lambda: sample_matrices(2, self.ranges)
        self.ranges = [[-1000, 1000]]
        self.torch_fn = torch.sub
        self.cpp_fn = lamppp.sub


class Mul(Operation):
    def __init__(self):
        super().__init__()
        self.atol = 1e-6
        self.sampler = lambda: sample_matrices(2, self.ranges)
        self.ranges = [[-1000, 1000]]
        self.torch_fn = torch.mul
        self.cpp_fn = lamppp.mul


class Div(Operation):
    def __init__(self):
        super().__init__()
        self.atol = 5e-6
        self.sampler = lambda: sample_matrices(2, self.ranges)
        self.ranges = [[-1000, -1e-3], [1e-3, 1000]]
        self.torch_fn = torch.div
        self.cpp_fn = lamppp.div


class Relu(Operation):
    def __init__(self):
        super().__init__()
        self.atol = 1e-6
        self.sampler = lambda: sample_matrices(1, self.ranges)
        self.ranges = [[-1000, 1000]]
        self.torch_fn = torch.relu
        self.cpp_fn = lamppp.relu


class Exp(Operation):
    def __init__(self):
        super().__init__()
        self.atol = 1e-4
        self.sampler = lambda: sample_matrices(1, self.ranges)
        self.ranges = [[-10, 10]]
        self.torch_fn = torch.exp
        self.cpp_fn = lamppp.exp


class Log(Operation):
    def __init__(self):
        super().__init__()
        self.atol = 1e-4
        self.sampler = lambda: sample_matrices(1, self.ranges)
        self.ranges = [[1e-3, 1000]]
        self.torch_fn = torch.log
        self.cpp_fn = lamppp.log


class Sqrt(Operation):
    def __init__(self):
        super().__init__()
        self.atol = 1e-6
        self.sampler = lambda: sample_matrices(1, self.ranges)
        self.ranges = [[1e-3, 1000]]
        self.torch_fn = torch.sqrt
        self.cpp_fn = lamppp.sqrt


class Abs(Operation):
    def __init__(self):
        super().__init__()
        self.atol = 1e-6
        self.sampler = lambda: sample_matrices(1, self.ranges)
        self.ranges = [[-1000, 1000]]
        self.torch_fn = torch.abs
        self.cpp_fn = lamppp.abs


class Sin(Operation):
    def __init__(self):
        super().__init__()
        self.atol = 1e-6
        self.sampler = lambda: sample_matrices(1, self.ranges)
        self.ranges = [[-10, 10]]
        self.torch_fn = torch.sin
        self.cpp_fn = lamppp.sin


class Cos(Operation):
    def __init__(self):
        super().__init__()
        self.atol = 1e-6
        self.sampler = lambda: sample_matrices(1, self.ranges)
        self.ranges = [[-10, 10]]
        self.torch_fn = torch.cos
        self.cpp_fn = lamppp.cos


class Tan(Operation):
    def __init__(self):
        super().__init__()
        self.atol = 1e-5
        self.sampler = lambda: sample_matrices(1, self.ranges)
        self.ranges = [[-1.5, 1.5]]
        self.torch_fn = torch.tan
        self.cpp_fn = lamppp.tan


class Clamp(Operation):
    def __init__(self, min_val, max_val):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.atol = 1e-6
        self.sampler = lambda: sample_matrices(1, self.ranges)
        self.ranges = [[-1000, 1000]]
        self.torch_fn = lambda t: torch.clamp(t, min=self.min_val, max=self.max_val)
        self.cpp_fn = lambda t: lamppp.clamp(t, self.min_val, self.max_val)


class Matmul(Operation):
    def __init__(self):
        super().__init__()
        self.atol = 3e-6
        self.sampler = lambda: sample_matrices(2, self.ranges)
        self.ranges = [[-100, 100]]
        self.torch_fn = torch.matmul
        self.cpp_fn = lamppp.matmul


class Transpose(Operation):
    def __init__(self):
        super().__init__()
        self.atol = 1e-6
        self.sampler = lambda: sample_matrices(1, self.ranges)
        self.ranges = [[-1000, 1000]]
        self.torch_fn = lambda x: x.T
        self.cpp_fn = lamppp.transpose


class Sum(Operation):
    def __init__(self, axis):
        super().__init__()
        self.atol = 1e-6
        self.sampler = lambda: sample_matrices(1, self.ranges)
        self.ranges = [[-1000, 1000]]
        self.torch_fn = lambda t: torch.sum(t, dim=axis)
        self.cpp_fn = lambda t: lamppp.sum(t, axis)
