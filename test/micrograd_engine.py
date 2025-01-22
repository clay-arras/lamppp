# adapted and extended from Karpathy's micrograd
import math


class Value:
    def __init__(self, data, _children=(), _op="", label="") -> None:
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data + other.data, _children=(self, other), _op="+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data * other.data, _children=(self, other), _op="*")

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return -1.0 * self

    def exp(self):
        out = Value(math.exp(self.data), _children=(self,), _op="exp")

        def _backward():
            self.grad += out.grad * out.data

        out._backward = _backward
        return out
    
    def log(self):
        out = Value(math.log(self.data), _children=(self,), _op="exp")

        def _backward():
            self.grad += out.grad * (1.0 / self.data)

        out._backward = _backward
        return out


    def __pow__(self, other):  # self ** other
        if not (isinstance(other, int) or isinstance(other, float)):
            assert False
        out = Value(self.data**other, _children=(self,), _op="**")

        def _backward():
            self.grad += out.grad * (other * (self.data ** (other - 1.0)))

        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + other.__neg__()

    def __truediv__(self, other):  # self/other
        return self * (other ** (-1.0))

    def tanh(self):
        e = (2 * self).exp()
        return (e - 1) / (e + 1)

    def relu(self):
        out = Value(self.data * (self.data > 0), _children=(self,), _op="relu")

        def _backward():
            self.grad += out.grad * (self.data > 0)

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
            return

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
