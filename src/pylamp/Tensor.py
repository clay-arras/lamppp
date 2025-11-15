import functools
from pylamp._C import *

@functools.total_ordering
class Tensor(_Variable):
    def __init__(self) -> None:
        pass

    def __add__(self, value: object, /) -> "Tensor":
        pass

    def __radd__(self, value: object, /) -> "Tensor":
        pass

    def __sub__(self, value: object, /) -> "Tensor":
        pass

    def __rsub__(self, value: object, /) -> "Tensor":
        pass

    def __mul__(self, value: object, /) -> "Tensor":
        pass

    def __rmul__(self, value: object, /) -> "Tensor":
        pass

    def __truediv__(self, value: object, /) -> "Tensor":
        pass

    def __rtruediv__(self, value: object, /) -> "Tensor":
        pass

    def __pow__(self, value: object, /) -> "Tensor":
        pass

    def __matmul__(self, value: object, /) -> "Tensor":
        pass

    def __rmatmul__(self, value: object, /) -> "Tensor":
        pass

    def __neg__(self) -> "Tensor":
        pass

    def __abs__(self) -> "Tensor":
        pass

    def __eq__(self, value: object, /) -> "Tensor":
        pass
    
    def __lt__(self, value: object, /) -> "Tensor":
        pass
        