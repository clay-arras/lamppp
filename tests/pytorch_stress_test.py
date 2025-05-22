import sys, os
from pathlib import Path

def get_project_root(marker = "pyproject.toml"):
    path = Path(__file__).resolve()
    for parent in [path, *path.parents]:
        if (parent / marker).is_file():
            return parent
    raise FileNotFoundError(f"Couldn’t locate {marker} in any parent directory.")

PROJECT_ROOT = get_project_root()
sys.path.append(os.path.join(PROJECT_ROOT, "build"))

import torch
import lamppp
import pytest
from operations_helper import *

ITERATIONS = 1000
EPSILON = 1e-10
TORCH_DTYPE = torch.float64


@pytest.fixture
def set_dtype(torch_dtype=TORCH_DTYPE):
    torch.set_default_dtype(TORCH_DTYPE)


@pytest.fixture
def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _to_row_major(mat):
    return torch.tensor(mat).flatten().tolist()


def _from_row_major(flat, like):
    t = torch.tensor(flat).reshape(torch.tensor(like).shape)
    return t.tolist()


def _to_lamppp_var(mat, requires_grad=True):
    ten = lamppp.cTensor(_to_row_major(mat), list(torch.tensor(mat).shape))
    return lamppp.cVariable(ten, requires_grad)


def _atol(pred, true):
    return float(torch.max(torch.abs(torch.tensor(pred) - torch.tensor(true))))


def _rtol(pred, true):
    return float(torch.max(torch.abs(torch.tensor(pred) - torch.tensor(true)))) / (
        float(torch.max(torch.tensor(true))) + EPSILON
    )


def compute_grads(lamppp_op, torch_op, mats):
    torch_vars = [torch.tensor(m, dtype=TORCH_DTYPE, requires_grad=True) for m in mats]
    torch_out = torch_op(*torch_vars)
    torch_out.backward(torch.ones_like(torch_out, dtype=TORCH_DTYPE))
    torch_vals = {
        "grads": [v.grad.tolist() for v in torch_vars],
        "out": [torch_out.data.tolist()],
    }

    lamppp_vars = [_to_lamppp_var(m) for m in mats]
    lamppp_out = lamppp_op(*lamppp_vars)
    lamppp_out.backward()
    lamppp_vals = {
        "grads": [_from_row_major(v.grad.data, m) for v, m in zip(lamppp_vars, mats)],
        "out": [_from_row_major(lamppp_out.data.data, torch_out.data.tolist())],
    }
    return lamppp_vals, torch_vals


def calculate_pair_tolerances(cg, tg):
    return _atol(cg, tg), _rtol(cg, tg)


@pytest.mark.usefixtures("set_seed", "set_dtype")
@pytest.mark.parametrize(
    "case",
    [
        Add,
        Sub,
        Mul,
        Div,
        Exp,
        Log,
        Sqrt,
        Abs,
        Sin,
        Cos,
        Tan,
        lambda: Clamp(-20, 20),  # todo: randomize this
        Matmul,
        Transpose,
        lambda: Sum(axis=0),
        lambda: Sum(axis=1),
        lambda: Max(axis=0),
        lambda: Max(axis=1),
        lambda: Min(axis=0),
        lambda: Min(axis=1),
    ],
    ids=[
        "add",
        "sub",
        "mul",
        "div",
        "exp",
        "log",
        "sqrt",
        "abs",
        "sin",
        "cos",
        "tan",
        "clamp",
        "matmul",
        "transpose",
        "sum_axis_0",
        "sum_axis_1",
        "max_axis_0",
        "max_axis_1",
        "min_axis_0",
        "min_axis_1",
    ],
)
def test_ops(case):
    instance = case()

    max_atol_forward, max_rtol_forward = 0, 0
    max_atol_backward, max_rtol_backward = 0, 0

    for _ in range(ITERATIONS):
        mats = instance.sampler()
        cpp_results, torch_results = compute_grads(
            instance.cpp_fn, instance.torch_fn, mats
        )

        for cpp_out, torch_out in zip(cpp_results["out"], torch_results["out"]):
            max_atol_forward, max_rtol_forward = (
                max(x, y)
                for x, y in zip(
                    (max_atol_forward, max_rtol_forward),
                    calculate_pair_tolerances(cpp_out, torch_out),
                )
            )
        for cpp_grad, torch_grad in zip(cpp_results["grads"], torch_results["grads"]):
            max_atol_backward, max_rtol_backward = (
                max(x, y)
                for x, y in zip(
                    (max_atol_backward, max_rtol_backward),
                    calculate_pair_tolerances(cpp_grad, torch_grad),
                )
            )

    atol_forward_pass = max_atol_forward <= instance.atol
    rtol_forward_pass = max_rtol_forward <= instance.rtol
    atol_backward_pass = (
        max_atol_backward <= instance.atol * instance.backward_atol_mult
    )
    rtol_backward_pass = (
        max_rtol_backward <= instance.rtol * instance.backward_atol_mult
    )

    assert atol_forward_pass
    assert rtol_forward_pass
    assert atol_backward_pass
    assert rtol_backward_pass
