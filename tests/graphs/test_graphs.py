"""Chain-level equivalence tests for the IR graph + kernel fusion module.

Each (template, seed) samples a slot fill once, then draws randomized inputs a
handful of times. A chain exercises many ops at once, so a few draws per fill is
enough -- far fewer iterations than the single-op stress suite needs.

Fusion on/off is an external flag; this file is agnostic to it. Run it once with
fusion disabled (validates the eager IR) and once enabled (validates that fusion
preserves numerics). A failure only in the enabled run is a fusion bug, already
reduced to a minimal chain.
"""

import numpy as np
import pytest
import pylamp

from templates import TEMPLATES
from ops import sample_fills
from runner import run_once

DRAWS = 4
SEEDS = list(range(16))


def _cuda_available():
    try:
        pylamp.Tensor(
            [[0.0]], device=pylamp.device.cuda, dtype=pylamp.dtype.float64
        )
        return True
    except Exception:
        return False


def _devices():
    devices = {"cpu": pylamp.device.cpu}
    if _cuda_available():
        devices["cuda"] = pylamp.device.cuda
    return devices


@pytest.mark.parametrize("device", _devices().values(), ids=_devices().keys())
@pytest.mark.parametrize("template", TEMPLATES, ids=[t.name for t in TEMPLATES])
@pytest.mark.parametrize("seed", SEEDS)
def test_graph(template, device, seed):
    rng = np.random.default_rng(seed)
    fills = sample_fills(template, rng)
    for _ in range(DRAWS):
        run_once(template, fills, device, rng)
