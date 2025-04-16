# %%
import sys, random, functools
sys.path.append("./build")
from cpp_custom_bind import *          # C++ bindings
import torch

# ───────────────────────────────────────────
#  Helpers / abstractions
# ───────────────────────────────────────────

EPS = 1e-5
_LO_N, _HI_N = -10, 10
_LO_R, _HI_R = 4, 10                # rows / cols for square‑ish mats

def rand_matrix(rows, cols, lo=_LO_N, hi=_HI_N):
    return [[random.uniform(lo, hi) for _ in range(cols)] for _ in range(rows)]

def rand_shape(min_r=_LO_R, max_r=_HI_R, min_c=_LO_R, max_c=_HI_R):
    """Returns a random (rows, cols) tuple."""
    return random.randint(min_r, max_r), random.randint(min_c, max_c)

def is_close(a, b, eps=EPS):
    return torch.all(torch.abs(torch.tensor(a) - torch.tensor(b)) < eps)

# Column‑major helpers
def _to_col_major(mat):
    return torch.tensor(mat).T.flatten().tolist()

def _from_col_major(flat, like):
    t = torch.tensor(flat).reshape(torch.tensor(like).T.shape).T
    return t.tolist()

def make_cpp_var(mat, requires_grad=True):
    ten = cTensor(_to_col_major(mat), list(torch.tensor(mat).shape))
    return cVariable(ten, requires_grad)

# Generic grad harness
def compute_grads(cpp_op, torch_op, mats, *extra):
    """
    mats: list of Python matrices (same length as operand count).
    Returns two lists → [cpp_grads], [torch_grads]
    """
    # ------  PyTorch path
    torch_vars = [torch.tensor(m, dtype=torch.float64, requires_grad=True) for m in mats]
    torch_out = torch_op(*torch_vars, *extra)
    torch_out.backward(torch.ones_like(torch_out, dtype=torch.float64))
    torch_grads = [v.grad.tolist() for v in torch_vars]

    # ------  C++ path
    cpp_vars = [make_cpp_var(m) for m in mats]
    cpp_out = cpp_op(*cpp_vars, *extra)   # noqa: F841 (just for side‑effects)
    cpp_grads = [_from_col_major(v.grad.data, m) for v, m in zip(cpp_vars, mats)]

    return cpp_grads, torch_grads

def run_test(name, cpp_op, torch_op, shape_fns, *extra):
    """
    shape_fns: iterable of callables that return shapes for each operand.
    extra:   extra positional args forwarded to the ops (e.g. axis for reduction).
    """
    mats = [rand_matrix(*shape_fn()) for shape_fn in shape_fns]

    cpp_grads, torch_grads = compute_grads(cpp_op, torch_op, mats, *extra)
    for i, (cg, tg) in enumerate(zip(cpp_grads, torch_grads)):
        assert is_close(cg, tg), (
            f"{name}: grad mismatch for input {i}\n"
            f"cpp : {cg}\n"
            f"torch: {tg}"
        )

# ───────────────────────────────────────────
#  Shape helpers for each arity / op family
# ───────────────────────────────────────────

same_shape = lambda: rand_shape()            # for element‑wise binary ops
unary_shape = lambda: rand_shape()
def matmul_shapes():
    m, k, n = random.randint(4, 8), random.randint(4, 8), random.randint(4, 8)
    return (m, k), (k, n)

# ───────────────────────────────────────────
#  Test case registry
# ───────────────────────────────────────────

TEST_CASES = [
    # ---------- existing binary element‑wise ops
    ("add",      add,      torch.add,     [same_shape, same_shape]),
    ("sub",      sub,      torch.sub,     [same_shape, same_shape]),
    ("mul",      mul,      torch.mul,     [same_shape, same_shape]),
    ("div",      div,      torch.div,     [same_shape, same_shape]),

    # ---------- NEW unary ops
    ("relu",     relu,     torch.relu,    [unary_shape]),
    ("exp",      exp,      torch.exp,     [unary_shape]),
    ("log",      log,      torch.log,     [unary_shape]),

    # ---------- NEW matrix ops
    ("matmul",   matmul,   torch.matmul,  [lambda: matmul_shapes()[0], lambda: matmul_shapes()[1]]),
    ("transpose", transpose,
                 lambda x: x.T,           # torch op for transpose
                 [unary_shape]),

    # ---------- NEW reduction ops
    # We choose a random explicit axis (not -1) per run
    # ("sum_axis0",
    #     lambda x, axis: sum(x, axis),     # cpp op: assume `sum(var, axis)`
    #     lambda t, axis: torch.sum(t, dim=axis),
    #     [unary_shape], 0  ),
    # ("sum_axis1",
    #     lambda x, axis: sum(x, axis),
    #     lambda t, axis: torch.sum(t, dim=axis),
    #     [unary_shape], 1  ),
]

# ───────────────────────────────────────────
#  Execute test suite
# ───────────────────────────────────────────

for case in TEST_CASES:
    name, cpp_op, torch_op, shape_fns, *extra = case
    run_test(name, cpp_op, torch_op, shape_fns, *extra)
print("✓ all gradient checks passed")

# %%
