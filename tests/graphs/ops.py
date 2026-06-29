"""Op tables, slot menus, and input scaling for the graph equivalence tests.

Each slot menu maps a name to a ``(torch_fn, lamp_fn)`` pair indexed by backend.
A template's slots are filled from these menus; the same fill drives both the
torch reference graph and the pylamp graph so the two stay structurally
identical. Kernel fusion only covers binary elementwise ops, so the binary menu
is the fusion target; unary / matmul / reduct are barriers that split a fused
region (and double as IR-correctness coverage).
"""

import numpy as np
import torch
import pylamp

BACKEND_TORCH = 0
BACKEND_LAMP = 1

# name -> (torch_fn, lamp_fn), each callable as f(a, b). Fusion target.
# div is intentionally omitted: its near-zero-denominator hazard needs a guard
# that doesn't belong in a generic same-shape elementwise slot.
BINARY = {
    "add": (torch.add, pylamp.add),
    "sub": (torch.sub, pylamp.sub),
    "mul": (torch.mul, pylamp.mul),
}

# name -> (torch_fn, lamp_fn), each callable as f(x). Trig only: total over the
# reals, so a chain can never wander out of the domain (no clamp guards needed).
UNARY = {
    "sin": (torch.sin, pylamp.sin),
    "cos": (torch.cos, pylamp.cos),
}

# name -> (torch_fn, lamp_fn), each callable as f(t, axis). pylamp reductions
# keep the reduced dim; mirror that with keepdim=True on the torch side so the
# two outputs line up.
REDUCT = {
    "sum": (lambda t, axis: torch.sum(t, dim=axis, keepdim=True), pylamp.sum),
    "max": (lambda t, axis: torch.max(t, dim=axis, keepdim=True).values, pylamp.max),
    "min": (lambda t, axis: torch.min(t, dim=axis, keepdim=True).values, pylamp.min),
}

# Fixed barrier, not a slot.
MATMUL = (torch.matmul, pylamp.matmul)

CATEGORIES = {"BIN": BINARY, "UNARY": UNARY, "REDUCT": REDUCT}


def row_normalize(arr):
    """Scale each row by its max-abs so values land in [-1, 1].

    This is pure data preprocessing applied before a leaf tensor is built, so it
    adds no graph nodes and doesn't perturb gradients -- both backends receive
    byte-identical leaves. Keeping inputs well-conditioned stops a deep mul chain
    from collapsing toward zero or a sum chain from drifting large.
    """
    arr = np.asarray(arr, dtype=np.float64)
    flat = arr.reshape(-1, arr.shape[-1])
    denom = np.abs(flat).max(axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return (flat / denom).reshape(arr.shape)


def sample_fills(template, rng):
    """Pick a concrete op for every slot, grouped by category in slot order.

    The returned dict maps a category to the list of ops chosen for that
    category's slots, in the order the template consumes them.
    """
    fills = {}
    for category in template.slots:
        names = list(CATEGORIES[category].keys())
        fills.setdefault(category, []).append(names[int(rng.integers(len(names)))])
    return fills


class OpSet:
    """Slot ops resolved for a single backend, grouped by category.

    A template's ``build`` reads ``ops.bin[i]`` / ``ops.unary[i]`` /
    ``ops.reduct[i]`` (slot order within each category) plus the fixed
    ``ops.matmul``.
    """

    def __init__(self, backend, fills):
        self.bin = [BINARY[n][backend] for n in fills.get("BIN", [])]
        self.unary = [UNARY[n][backend] for n in fills.get("UNARY", [])]
        self.reduct = [REDUCT[n][backend] for n in fills.get("REDUCT", [])]
        self.matmul = MATMUL[backend]
