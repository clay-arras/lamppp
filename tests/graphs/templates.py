"""Hand-authored chain skeletons with swappable, typed op slots.

A template is shape-agnostic wiring: it never constructs tensors or assumes
equal shapes, it only threads roles together. Shapes live in ``input_shapes``
(external) so introducing broadcasting later is a shape swap, not a body change.
The runner samples a concrete op for each slot and feeds leaves in.
"""

from dataclasses import dataclass
from typing import Callable

S = (16, 16)  # square block, fully fusible elementwise
M = (16, 8)   # post-matmul / right-operand block


@dataclass
class Template:
    name: str
    input_shapes: list  # one shape per leaf, in xs order
    slots: list         # category ("BIN"/"UNARY"/"REDUCT") per slot, in build order
    build: Callable     # build(ops, xs) -> head tensor


def _long_build(ops, xs):
    # Pure binary elementwise, same shape throughout: one maximal fused region.
    t = ops.bin[0](xs[0], xs[1])
    t = ops.bin[1](t, xs[2])
    t = ops.bin[2](t, xs[3])
    t = ops.bin[3](t, xs[4])
    t = ops.bin[4](t, xs[5])
    return t


long_fuse_chain = Template(
    name="long_fuse_chain",
    input_shapes=[S, S, S, S, S, S],
    slots=["BIN", "BIN", "BIN", "BIN", "BIN"],
    build=_long_build,
)


def _barrier_build(ops, xs):
    # Fusible runs separated by barriers (unary, matmul, reduction). Exercises
    # that the IR partitions a graph correctly around things fusion can't absorb.
    t = ops.bin[0](xs[0], xs[1])   # fusible run
    t = ops.bin[1](t, xs[2])
    t = ops.unary[0](t)            # barrier: unary
    t = ops.matmul(t, xs[3])       # barrier: matmul  (16,16)@(16,8) -> (16,8)
    t = ops.bin[2](t, xs[4])       # fusible run, new shape
    t = ops.reduct[0](t, 0)        # barrier: reduction -> (1,8)
    return t


barrier_chain = Template(
    name="barrier_chain",
    input_shapes=[S, S, S, M, M],
    slots=["BIN", "BIN", "UNARY", "BIN", "REDUCT"],
    build=_barrier_build,
)


TEMPLATES = [long_fuse_chain, barrier_chain]
