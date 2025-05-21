from hypothesis import given
from hypothesis.strategies import composite, lists, integers, sampled_from, permutations
import torch
import math

NUM_TENS = 5
DTYPES = [torch.float32, torch.float64]
OPS = [torch.add, torch.sub, torch.mul]


class DSU:
    def __init__(self, n: int, vars):
        self.e = [-1] * n
        self.list_head = vars

    def get(self, x: int) -> int:
        """Find with path compression."""
        if self.e[x] < 0:
            return x
        self.e[x] = self.get(self.e[x])
        return self.e[x]

    def get_head(self, x: int):
        return self.list_head[self.get(x)]

    def size(self, x: int) -> int:
        """Return size of the set containing x."""
        root = self.get(x)
        return -self.e[root]

    def unite(self, x: int, y: int, op) -> bool:
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
            self.list_head[x].shape, self.list_head[y].shape
        )
        self.list_head[x] = torch.reshape(self.list_head[x], nshape_i)
        self.list_head[y] = torch.reshape(self.list_head[y], nshape_j)

        self.list_head[x] = op(self.list_head[x], self.list_head[y])
        self.list_head[y] = None
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
            lists(integers(min_value=1, max_value=10), min_size=1, max_size=5),
            min_size=NUM_TENS,
            max_size=NUM_TENS,
        )
    )
    dtypes = draw(
        lists(
            sampled_from(DTYPES),
            min_size=NUM_TENS,
            max_size=NUM_TENS,
        )
    )
    return [torch.rand(shape, dtype=dtype) for shape, dtype in zip(shapes, dtypes)]


@given(
    tensors=build_tensors(),
    operations=lists(sampled_from(OPS), min_size=NUM_TENS - 1, max_size=NUM_TENS - 1),
    edges=permutations([[i, i + 1] for i in range(NUM_TENS - 1)]),
)
def test_graph_builder(tensors, operations, edges):
    assert len(tensors) == NUM_TENS and "test_graph_builder: died"

    graph = DSU(len(tensors), tensors)
    for edge in edges:
        if graph.get(edge[0]) == graph.get(edge[1]):
            assert False and "test_graph_builder: died"
        graph.unite(edge[0], edge[1], operations.pop())

    # graph.get_head(0)


# test_graph_builder()
