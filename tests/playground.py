import sys, os

PROJECT_ROOT = "/home/nlin/workspace/code/projects/autograd_cpp"
sys.path.append(os.path.join(PROJECT_ROOT, "build"))

from hypothesis import Verbosity, given, settings
from hypothesis.strategies import composite, lists, integers, sampled_from, permutations
import torch
import math
import lamppp


"""
TODO: 
- create more datatypes to test dtype promotion
- test transpose and matmul
- backpropogate the expected error (i.e. add with mul should propagate the two errors)
"""


NUM_TENS = 4
DTYPES = [torch.float64]  # bindings for Lamppp other dtypes not supported yet
OPS = [
    {"id": "add", "torch": torch.add, "lamp": lamppp.add},
    {"id": "sub", "torch": torch.sub, "lamp": lamppp.sub},
    {"id": "mul", "torch": torch.mul, "lamp": lamppp.mul},
    {
        "id": "div_clamp_y_1e-1_1e6",
        "torch": lambda x, y: torch.div(x, torch.clamp(y, 1e-1, 1e6)),
        "lamp": lambda x, y: lamppp.div(x, lamppp.clamp(y, 1e-1, 1e6)),
    },
]
UNARY_OPS = [
    {
        "id": "exp_clamp_-10_10",
        "torch": lambda x: torch.exp(torch.clamp(x, -10, 10)),
        "lamp": lambda x: lamppp.exp(lamppp.clamp(x, -10, 10)),
    },
    {
        "id": "log_clamp_1e-3_500",
        "torch": lambda x: torch.log(torch.clamp(x, 1e-3, 500)),
        "lamp": lambda x: lamppp.log(lamppp.clamp(x, 1e-3, 500)),
    },
    {
        "id": "sqrt_clamp_1e-3_500",
        "torch": lambda x: torch.sqrt(torch.clamp(x, 1e-3, 500)),
        "lamp": lambda x: lamppp.sqrt(lamppp.clamp(x, 1e-3, 500)),
    },
    {"id": "abs", "torch": torch.abs, "lamp": lamppp.abs},
    {"id": "sin", "torch": torch.sin, "lamp": lamppp.sin},
    {"id": "cos", "torch": torch.cos, "lamp": lamppp.cos},
    {
        "id": "sqrt_clamp_-1_1",
        "torch": lambda x: torch.sqrt(torch.clamp(x, -1, 1)),
        "lamp": lambda x: lamppp.sqrt(lamppp.clamp(x, -1, 1)),
    },
]
REDUCT_OPS = lambda axis: [
    {
        "id": f"sum_axis_{axis}",
        "torch": lambda x: torch.sum(x, dim=axis % x.ndim, keepdim=False),
        "lamp": lambda x: lamppp.squeeze(
            lamppp.sum(x, axis % len(x.data.shape)), axis % len(x.data.shape)
        ),
    },
    {
        "id": f"min_axis_{axis}",
        "torch": lambda x: torch.min(x, dim=axis % x.ndim, keepdim=False).values,
        "lamp": lambda x: lamppp.squeeze(
            lamppp.min(x, axis % len(x.data.shape)), axis % len(x.data.shape)
        ),
    },
    {
        "id": f"max_axis_{axis}",
        "torch": lambda x: torch.max(x, dim=axis % x.ndim, keepdim=False).values,
        "lamp": lambda x: lamppp.squeeze(
            lamppp.max(x, axis % len(x.data.shape)), axis % len(x.data.shape)
        ),
    },
]


class DSU:
    def __init__(self, n: int, torch_vars, lamp_vars):
        self.e = [-1] * n
        self.torch_head = list(torch_vars)
        self.lamp_head = list(lamp_vars)

    def get(self, x: int) -> int:
        """Find with path compression."""
        if self.e[x] < 0:
            return x
        self.e[x] = self.get(self.e[x])
        return self.e[x]

    def get_torch(self, x: int):
        return self.torch_head[self.get(x)]

    def get_lamp(self, x: int):
        return self.lamp_head[self.get(x)]

    def size(self, x: int) -> int:
        """Return size of the set containing x."""
        root = self.get(x)
        return -self.e[root]

    def unite(self, x: int, y: int, bin_op, una_op) -> bool:
        """Union by size. Returns True if merged, False if already same set."""
        x = self.get(x)
        y = self.get(y)
        if x == y:
            return False
        if self.e[x] > self.e[y]:
            x, y = y, x
        self.e[x] += self.e[y]
        self.e[y] = x

        nshape_i, nshape_j = find_common_reshape(
            self.torch_head[x].shape, self.torch_head[y].shape
        )

        self.torch_head[x] = torch.reshape(self.torch_head[x], nshape_i)
        self.torch_head[y] = torch.reshape(self.torch_head[y], nshape_j)

        self.lamp_head[x] = lamppp.reshape(self.lamp_head[x], nshape_i)
        self.lamp_head[y] = lamppp.reshape(self.lamp_head[y], nshape_j)

        self.torch_head[x] = bin_op["torch"](self.torch_head[x], self.torch_head[y])
        self.torch_head[x] = una_op["torch"](self.torch_head[x])
        self.torch_head[y] = None

        self.lamp_head[x] = bin_op["lamp"](self.lamp_head[x], self.lamp_head[y])
        self.lamp_head[x] = una_op["lamp"](self.lamp_head[x])
        self.lamp_head[y] = None

        return True


def find_common_reshape(init_a, init_b):
    a = list(init_a)
    b = list(init_b)

    while len(a) < len(b):
        a.insert(0, 1)
    while len(b) < len(a):
        b.insert(0, 1)

    N = len(a)
    a_res = 1
    b_res = 1
    for i in range(N - 1, -1, -1):
        if a[i] == 1 or b[i] == 1:
            continue
        else:
            a_res *= a[i]
            b_res *= b[i]
            gcd = math.gcd(a_res, b_res)
            a_res = int(a_res / gcd)
            b_res = int(b_res / gcd)
            a[i] = gcd
            b[i] = gcd

    if a_res != 1 or b_res != 1:
        a.insert(0, 1)
        b.insert(0, b_res)
        a.insert(0, a_res)

    return a, b


@composite
def build_tensors(draw):
    shapes = draw(
        lists(
            lists(integers(min_value=1, max_value=5), min_size=1, max_size=5),
            min_size=NUM_TENS,
            max_size=NUM_TENS,
        )
    )
    sizes = [torch.prod(torch.tensor(arr)).item() for arr in shapes]
    tens_vals = [
        draw(
            lists(integers(min_value=-10, max_value=10), min_size=sz, max_size=sz),
        )
        for sz in sizes
    ]
    dtypes = draw(
        lists(
            sampled_from(DTYPES),
            min_size=NUM_TENS,
            max_size=NUM_TENS,
        )
    )
    return {
        "bodies": tens_vals,
        "shapes": shapes,
        "dtypes": dtypes,
    }


@composite
def build_unaries(draw):
    srand_reduct = draw(
        lists(
            integers(min_value=1, max_value=5),
            min_size=NUM_TENS,
            max_size=NUM_TENS,
        )
    )
    srand_choice = draw(
        lists(
            integers(min_value=0, max_value=1),
            min_size=NUM_TENS - 1,
            max_size=NUM_TENS - 1,
        )
    )

    reduct_fns = []
    for i in srand_reduct:
        reduct_fns.extend(REDUCT_OPS(i))

    return [
        draw(sampled_from(reduct_fns)) if i else draw(sampled_from(UNARY_OPS))
        for i in srand_choice
    ]


@given(
    meta=build_tensors(),
    bin_ops=lists(sampled_from(OPS), min_size=NUM_TENS - 1, max_size=NUM_TENS - 1),
    una_ops=build_unaries(),
    edges=permutations([[i, i + 1] for i in range(NUM_TENS - 1)]),
)
@settings(deadline=400, verbosity=Verbosity.verbose, max_examples=20)
def test_graph_builder(meta, bin_ops, una_ops, edges):
    torch_tensors = [
        torch.tensor(
            meta["bodies"][i], dtype=meta["dtypes"][i], requires_grad=True
        ).reshape(meta["shapes"][i])
        for i in range(len(meta["bodies"]))
    ]
    lamp_tensors = [
        lamppp.cVariable(lamppp.cTensor(meta["bodies"][i], meta["shapes"][i]), True)
        for i in range(len(meta["bodies"]))
    ]

    graph = DSU(len(torch_tensors), torch_vars=torch_tensors, lamp_vars=lamp_tensors)
    for edge in edges:
        if graph.get(edge[0]) == graph.get(edge[1]):
            assert False and "test_graph_builder: died"
        graph.unite(edge[0], edge[1], bin_ops.pop(), una_ops.pop())

    graph.get_torch(0).backward(torch.ones_like(graph.get_torch(0)))
    graph.get_lamp(0).backward()

    for i in range(len(torch_tensors)):
        t_ten = torch_tensors[i]
        l_ten_var = lamp_tensors[i]

        assert (
            list(t_ten.shape) == l_ten_var.data.shape
        ), f"Tensor {i} shapes mismatch. PyTorch: {t_ten.shape}, Lamppp: {l_ten_var.data.shape}"

        if t_ten.grad is not None:
            assert (
                l_ten_var.grad is not None
            ), f"Tensor {i}: PyTorch has grad, but Lamppp grad is None."
            assert (
                l_ten_var.grad.data is not None
            ), f"Tensor {i}: PyTorch has grad, but Lamppp grad.data is None."

            lamp_grad_torch_compatible = torch.tensor(
                l_ten_var.grad.data, dtype=t_ten.dtype
            ).reshape(l_ten_var.data.shape)

            assert torch.allclose(
                t_ten.grad, lamp_grad_torch_compatible, rtol=1e-4
            ), f"Gradients for tensor {i} do not match.\nPyTorch grad: {t_ten.grad}\nLamppp grad: {lamp_grad_torch_compatible}"
        else:
            assert (
                l_ten_var.grad is None
                or l_ten_var.grad.data is None
                or not l_ten_var.grad.data.data
            ), f"Tensor {i}: PyTorch grad is None, but Lamppp grad is not effectively None.\nLamppp grad data: {l_ten_var.grad.data.data if l_ten_var.grad and l_ten_var.grad.data else 'N/A'}"
