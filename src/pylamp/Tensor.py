import functools
from math import prod
from pylamp._C import _Variable, _Tensor, dtype, device

Scalar = int | float


def _flatten_array(arr) -> tuple[list[Scalar], list[int]]:
    shape = []

    def _rec(x, depth=0):
        if isinstance(x, list):
            if len(shape) == depth:
                shape.append(len(x))
            elif shape[depth] != len(x):
                raise ValueError("Pylamp: array must be uniform")
            for i in x:
                yield from _rec(i, depth + 1)

        elif isinstance(x, Scalar):
            yield x
        else:
            raise ValueError("Pylamp: invalid input type")

    return list(_rec(arr)), shape


@functools.total_ordering
class Tensor(_Variable):
    def __init__(
        self,
        *args,
        requires_grad: bool = False,
        device: device = device.cpu,
        dtype: dtype = dtype.float32,
    ) -> None:
        if len(args) == 1 and isinstance(args[0], list):
            data, shape = _flatten_array(args[0])
            ten = _Tensor(data, shape, device, dtype)
            super().__init__(ten, requires_grad)
        elif len(args) >= 1 and all([isinstance(arg, Scalar) for arg in args]):
            # TODO: this will be slow
            ten = _Tensor([0] * prod(args), args, device, dtype)
            super().__init__(ten, requires_grad)

        else:
            raise ValueError("Pylamp: invalid Tensor constructor arguments")

    def __repr__(self) -> str:
        return super().__repr__()
