"""Op menus, slot sampling, and input scaling for the graph equivalence tests.

Each menu entry is a ``(torch_fn, lamp_fn)`` pair indexed by backend, so one
sampled fill drives both the torch reference graph and the rushlite graph.

Ops whose domain can blow up (div by ~0, neg base ^ frac, log/sqrt of <=0,
tan near pi/2, exp overflow) get a *separate* guarded def that wraps the
operand in ``clamp`` into a safe domain. The guard is identical on both
backends, so the differential check stays valid. Ops that are total over the
reals (add/sub/mul, neg/abs/sin/cos) are used raw.
"""

import numpy as np
import torch
import rushlite

BACKEND_TORCH = 0
BACKEND_LAMP = 1

EPS = 1e-3  # positive floor for log/sqrt domain
BIG = 1e3  # magnitude ceiling
EXP_HI = 10.0  # exp arg ceiling (e^10 ~ 2e4, no overflow)
TAN_LIM = 1.5  # < pi/2, keeps tan off its asymptotes
# div/pow are doubly ill-conditioned: a denom or base near zero blows up both
# the value and the gradient (pow's exponent grad ~ base^exp * ln(base)).
# A comfortable floor (not EPS) keeps the differentiable region O(1) so float
# noise stays well under tolerance.
DENOM_LO = 0.5  # div denominator floor
BASE_LO = 0.5  # pow base floor
BASE_HI = 2.0  # pow base ceiling
POW_LIM = 2.0  # pow exponent magnitude ceiling


# --- binary elementwise: the fusion target ---------------------------------

# Total ops, safe to chain raw -> a pure fusible region.
BINARY_SAFE = {
    "add": (torch.add, rushlite.add),
    "sub": (torch.sub, rushlite.sub),
    "mul": (torch.mul, rushlite.mul),
}

# Domain-bounded ops, guarded with clamp.
BINARY_BOUNDED = {
    "div": (
        lambda a, b: torch.div(a, torch.clamp(b, min=DENOM_LO, max=BIG)),
        lambda a, b: rushlite.div(a, rushlite.clamp(b, DENOM_LO, BIG)),
    ),
    "pow": (
        lambda a, b: torch.pow(
            torch.clamp(a, min=BASE_LO, max=BASE_HI),
            torch.clamp(b, min=-POW_LIM, max=POW_LIM),
        ),
        lambda a, b: rushlite.pow(
            rushlite.clamp(a, BASE_LO, BASE_HI), rushlite.clamp(b, -POW_LIM, POW_LIM)
        ),
    ),
}

# --- unary ------------------------------------------------------------------

UNARY_SAFE = {
    "neg": (torch.neg, rushlite.neg),
    "abs": (torch.abs, rushlite.abs),
    "sin": (torch.sin, rushlite.sin),
    "cos": (torch.cos, rushlite.cos),
}

UNARY_BOUNDED = {
    "exp": (
        lambda x: torch.exp(torch.clamp(x, min=-BIG, max=EXP_HI)),
        lambda x: rushlite.exp(rushlite.clamp(x, -BIG, EXP_HI)),
    ),
    "log": (
        lambda x: torch.log(torch.clamp(x, min=EPS, max=BIG)),
        lambda x: rushlite.log(rushlite.clamp(x, EPS, BIG)),
    ),
    "sqrt": (
        lambda x: torch.sqrt(torch.clamp(x, min=EPS, max=BIG)),
        lambda x: rushlite.sqrt(rushlite.clamp(x, EPS, BIG)),
    ),
    "tan": (
        lambda x: torch.tan(torch.clamp(x, min=-TAN_LIM, max=TAN_LIM)),
        lambda x: rushlite.tan(rushlite.clamp(x, -TAN_LIM, TAN_LIM)),
    ),
}

UNARY = {**UNARY_SAFE, **UNARY_BOUNDED}

# --- reduction (barrier) ----------------------------------------------------

# rushlite reductions keep the reduced dim; mirror with keepdim=True on torch.
REDUCT = {
    "sum": (lambda t, axis: torch.sum(t, dim=axis, keepdim=True), rushlite.sum),
    "max": (lambda t, axis: torch.max(t, dim=axis, keepdim=True).values, rushlite.max),
    "min": (lambda t, axis: torch.min(t, dim=axis, keepdim=True).values, rushlite.min),
}

# Fixed barrier, not a sampled slot.
MATMUL = (torch.matmul, rushlite.matmul)

CATEGORIES = {
    "BIN": BINARY_SAFE,
    "BIN_BOUNDED": BINARY_BOUNDED,
    "UNARY": UNARY,
    "REDUCT": REDUCT,
}


def row_normalize(arr):
    """Scale each row by its max-abs into [-1, 1].

    Pure preprocessing on the leaf array (no graph node), so both backends get
    byte-identical leaves and gradients are untouched. Keeps deep chains
    well-conditioned instead of collapsing to 0 or drifting large.
    """
    arr = np.asarray(arr, dtype=np.float64)
    flat = arr.reshape(-1, arr.shape[-1])
    denom = np.abs(flat).max(axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return (flat / denom).reshape(arr.shape)


def sample_fills(template, rng):
    """Pick one op per slot, grouped by category in slot order."""
    fills = {}
    for category in template.slots:
        names = list(CATEGORIES[category].keys())
        fills.setdefault(category, []).append(names[int(rng.integers(len(names)))])
    return fills


class OpSet:
    """Slot ops resolved for one backend; a template reads them by category."""

    def __init__(self, backend, fills):
        self.bin = [BINARY_SAFE[n][backend] for n in fills.get("BIN", [])]
        self.bin_bounded = [
            BINARY_BOUNDED[n][backend] for n in fills.get("BIN_BOUNDED", [])
        ]
        self.unary = [UNARY[n][backend] for n in fills.get("UNARY", [])]
        self.reduct = [REDUCT[n][backend] for n in fills.get("REDUCT", [])]
        self.matmul = MATMUL[backend]
