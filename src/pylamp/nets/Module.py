from abc import ABC, abstractmethod
from typing import Any, Generator

from pylamp.Tensor import Tensor


class Module(ABC):
    def __init__(self) -> None:
        self._params_dict = {}

    @abstractmethod
    def forward(self, *args: Any, **kwds: Any) -> Any:
        pass

    def __setattr__(self, name: str, value: Any, /) -> None:
        if isinstance(value, Module) or isinstance(value, Tensor):
            self._params_dict[name] = value
        super().__setattr__(name, value)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def parameters(self) -> Generator[Tensor, None, None]:
        for val in self._params_dict.values():
            if isinstance(val, Tensor):
                yield val
            elif isinstance(val, Module):
                yield from val.parameters()
            else:
                raise Exception()

    def named_parameters(
        self, _path: str = []
    ) -> Generator[tuple[str, Tensor], None, None]:
        for key, val in self._params_dict.items():
            if isinstance(val, Tensor):
                yield (".".join(map(str, (_path + [key]))), val)
            elif isinstance(val, Module):
                yield from val.named_parameters(_path=_path + [key])
            else:
                raise Exception()

    def zero_grad(self) -> None:  # TODO
        pass

    def to(self) -> None:  # TODO
        pass

    def __repr__(self) -> str:  # TODO
        return super().__repr__()
