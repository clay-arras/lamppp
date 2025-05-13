import sys, os

PROJECT_ROOT = "/home/nlin/workspace/code/projects/autograd_cpp"
sys.path.append(os.path.join(PROJECT_ROOT, "build"))

import torch
import lamppp
from operations_helper import (
    Add,
    Sub,
    Mul,
    Div,
    Relu,
    Exp,
    Log,
    Sqrt,
    Abs,
    Sin,
    Cos,
    Tan,
    Clamp,
    Matmul,
    Sum,
    Transpose,
)

ITERATIONS = 1000
EPSILON = 1e-10
TORCH_DTYPE = torch.float64

torch.set_default_dtype(TORCH_DTYPE)


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 42
set_seed(seed)


def compute_grads(cpp_op, torch_op, mats):
    def _to_row_major(mat):
        return torch.tensor(mat).flatten().tolist()

    def _from_row_major(flat, like):
        t = torch.tensor(flat).reshape(torch.tensor(like).shape)
        return t.tolist()

    def _to_cpp_var(mat, requires_grad=True):
        ten = lamppp.cTensor(_to_row_major(mat), list(torch.tensor(mat).shape))
        return lamppp.cVariable(ten, requires_grad)

    torch_vars = [torch.tensor(m, dtype=TORCH_DTYPE, requires_grad=True) for m in mats]
    torch_out = torch_op(*torch_vars)
    torch_out.backward(torch.ones_like(torch_out, dtype=TORCH_DTYPE))
    torch_vals = {
        "grads": [v.grad.tolist() for v in torch_vars],
        "out": [torch_out.data.tolist()],
    }

    cpp_vars = [_to_cpp_var(m) for m in mats]
    cpp_out = cpp_op(*cpp_vars)
    cpp_vals = {
        "grads": [_from_row_major(v.grad.data, m) for v, m in zip(cpp_vars, mats)],
        "out": [_from_row_major(cpp_out.data.data, torch_out.data.tolist())],
    }

    return cpp_vals, torch_vals


def run_test(case, its):
    def _atol(pred, true):
        return float(torch.max(torch.abs(torch.tensor(pred) - torch.tensor(true))))

    def _rtol(pred, true):
        return float(
            torch.max(torch.abs(torch.tensor(pred) - torch.tensor(true)))
        ) / (float(torch.max(torch.tensor(true))) + EPSILON)

    max_atol_forward, max_rtol_forward = 0, 0
    max_atol_backward, max_rtol_backward = 0, 0

    def check_tolerances(cg, tg, pass_type):
        nonlocal max_atol_forward, max_rtol_forward, max_atol_backward, max_rtol_backward
        atol_ = _atol(cg, tg)
        rtol_ = _rtol(cg, tg)

        if pass_type == "forward":
            max_atol_forward = max(max_atol_forward, atol_)
            max_rtol_forward = max(max_rtol_forward, rtol_)
        elif pass_type == "backward":
            max_atol_backward = max(max_atol_backward, atol_)
            max_rtol_backward = max(max_rtol_backward, rtol_)
        else:
            raise ValueError(f"Unknown pass_type: {pass_type}")

    for i in range(its):
        mats = case.sampler()
        cpp_results, torch_results = compute_grads(case.cpp_fn, case.torch_fn, mats)

        for cpp_out, torch_out in zip(cpp_results["out"], torch_results["out"]):
            check_tolerances(cpp_out, torch_out, "forward")
        for cpp_grad, torch_grad in zip(cpp_results["grads"], torch_results["grads"]):
            check_tolerances(cpp_grad, torch_grad, "backward")

    forward_atol_threshold = case.atol
    forward_rtol_threshold = case.rtol
    backward_atol_threshold = case.atol * case.backward_atol_mult
    backward_rtol_threshold = case.rtol * case.backward_atol_mult

    atol_forward_pass = max_atol_forward <= forward_atol_threshold
    rtol_forward_pass = max_rtol_forward <= forward_rtol_threshold
    atol_backward_pass = max_atol_backward <= backward_atol_threshold
    rtol_backward_pass = max_rtol_backward <= backward_rtol_threshold

    atol_forward_result = "✅ pass" if atol_forward_pass else "❌ fail"
    rtol_forward_result = "✅ pass" if rtol_forward_pass else "❌ fail"
    atol_backward_result = "✅ pass" if atol_backward_pass else "❌ fail"
    rtol_backward_result = "✅ pass" if rtol_backward_pass else "❌ fail"

    print(
        f"{case.__class__.__name__}:\n"
        f"  Forward : atol {atol_forward_result} (max={max_atol_forward:.3e}, thr={forward_atol_threshold:.1e}), "
        f"rtol {rtol_forward_result} (max={max_rtol_forward:.3e}, thr={forward_rtol_threshold:.1e})\n"
        f"  Backward: atol {atol_backward_result} (max={max_atol_backward:.3e}, thr={backward_atol_threshold:.1e}), "
        f"rtol {rtol_backward_result} (max={max_rtol_backward:.3e}, thr={backward_rtol_threshold:.1e})"
    )


def main():
    OPERATIONS = [
        Add,
        Sub,
        Mul,
        Div,
        Relu,
        Exp,
        Log,
        Sqrt,
        Abs,
        Sin,
        Cos,
        Tan,
        lambda: Clamp(-20, 20), # todo: randomize this
        Matmul,
        Transpose,
        lambda: Sum(axis=0),
        lambda: Sum(axis=1),
    ]

    for case in OPERATIONS:
        run_test(case(), ITERATIONS)


if __name__ == "__main__":
    main()
