import os

import torch
import pylamp
import pytest
from operations import *
from testutils import *

ITERATIONS = 1000
EPSILON = 1e-10
TORCH_DTYPE = torch.float64


def get_case():
    return {
        "add": Add,
        "sub": Sub,
        "mul": Mul,
        "div": Div,
        "exp": Exp,
        "log": Log,
        "sqrt": Sqrt,
        "abs": Abs,
        "sin": Sin,
        "cos": Cos,
        "tan": Tan,
        "clamp": lambda: Clamp(-20, 20),  # todo: randomize this
        "matmul": Matmul,
        "transpose": Transpose,
        "sum_axis_0": lambda: Sum(axis=0),
        "sum_axis_1": lambda: Sum(axis=1),
        "max_axis_0": lambda: Max(axis=0),
        "max_axis_1": lambda: Max(axis=1),
        "min_axis_0": lambda: Min(axis=0),
        "min_axis_1": lambda: Min(axis=1),
    }


def get_device():
    devices = {"cpu": pylamp.device.cpu}
    if torch.cuda.is_available():  # TODO: this isn't great, should add pylamp option
        devices.update({"cuda": pylamp.device.cuda})
    return devices


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


def compute_grads(pylamp_op, torch_op, mats, device):
    torch_vars = [torch.tensor(m, dtype=TORCH_DTYPE, requires_grad=True) for m in mats]
    torch_out = torch_op(*torch_vars)
    torch_out.backward(torch.ones_like(torch_out, dtype=TORCH_DTYPE))
    torch_vals = {
        "grads": [v.grad.tolist() for v in torch_vars],
        "out": [torch_out.data.tolist()],
    }

    pylamp_vars = [to_pylamp_var(m, device) for m in mats]
    pylamp_out = pylamp_op(*pylamp_vars)
    pylamp_out.backward()
    pylamp_vals = {
        "grads": [
            from_row_major(v.grad.tolist(), m) for v, m in zip(pylamp_vars, mats)
        ],
        "out": [from_row_major(pylamp_out.tolist(), torch_out.data.tolist())],
    }
    return pylamp_vals, torch_vals


@pytest.mark.usefixtures("set_seed", "set_dtype")
@pytest.mark.parametrize("case", get_case().values(), ids=get_case().keys())
@pytest.mark.parametrize("device", get_device().values(), ids=get_device().keys())
def test_ops(case, device):
    instance = case()

    max_atol_forward, max_rtol_forward = 0, 0
    max_atol_backward, max_rtol_backward = 0, 0

    for _ in range(ITERATIONS):
        mats = instance.sampler()
        cpp_results, torch_results = compute_grads(
            instance.cpp_fn, instance.torch_fn, mats, device
        )

        for cpp_out, torch_out in zip(cpp_results["out"], torch_results["out"]):
            max_atol_forward, max_rtol_forward = (
                max(x, y)
                for x, y in zip(
                    (max_atol_forward, max_rtol_forward),
                    calculate_pair_tolerances(cpp_out, torch_out, EPSILON),
                )
            )
        for cpp_grad, torch_grad in zip(cpp_results["grads"], torch_results["grads"]):
            max_atol_backward, max_rtol_backward = (
                max(x, y)
                for x, y in zip(
                    (max_atol_backward, max_rtol_backward),
                    calculate_pair_tolerances(cpp_grad, torch_grad, EPSILON),
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
