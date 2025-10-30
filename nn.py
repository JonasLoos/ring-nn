from abc import ABC, abstractmethod
import pickle
from typing import Callable, Any

from tensor import Tensor, RingTensor


class Partial:
    """Wrappper for a function with fixed arguments and nice repr."""
    def __init__(self, fn: Callable, *args, **kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        return self.fn(x, *self.args, **self.kwargs)

    def __repr__(self):
        args_repr = ', '.join([f"{arg!r}" for arg in self.args] + [f"{k}={v!r}" for k, v in self.kwargs.items()])
        return f"{self.fn.__name__}({args_repr})"


class Model(ABC):
    weights: list[Tensor]
    nparams: int

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str):
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str):
        raise NotImplementedError


class Module(Model):
    _weights: list[Tensor] = []

    @property
    def weights(self) -> list[Tensor]:
        all_weights = self._weights or []
        for m in self.__dict__.values():
            if isinstance(m, Module):
                all_weights += m.weights
        return all_weights

    @property
    def nparams(self) -> int:
        return sum(w.size for w in self.weights)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.weights, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            for old, new in zip(self.weights, pickle.load(f)):
                old.set_to(new)
        return self


class RingFF(Module):
    def __init__(self, input_size: int, output_size: int):
        self._weights = [RingTensor.rand((input_size, output_size), requires_grad=True)]
        # We don't need any bias, because it would be mathematically equivalent to just shifting the weights of the following layer by the corresponding amount.

    def forward(self, x: RingTensor) -> RingTensor:
        return (x.unsqueeze(-1) - self._weights[0]).cos().mean(axis=-2)
        # return (x.unsqueeze(-1) - self._weights[0]).poly_sigmoid(1.2, 4).mean(axis=-2)

    def __repr__(self):
        return f"RingFF(input_size={self._weights[0].shape[0]}, output_size={self._weights[0].shape[1]})"


class RingConv(Module):
    def __init__(self, input_size: int, output_size: int, window_size: int = 3, padding: int = 1, stride: int = 1):
        self.window_size = window_size
        self.padding = padding
        self.stride = stride
        self._weights = [RingTensor.rand((input_size, window_size, window_size, output_size), requires_grad=True)]

    def forward(self, x: RingTensor) -> RingTensor:
        return (x.sliding_window_2d(self.window_size, self.padding, self.stride).unsqueeze(-1) - self._weights[0]).cos().mean(axis=(-4,-3,-2))
        # return (x.sliding_window_2d(self.window_size, self.padding, self.stride).unsqueeze(-1) - self._weights[0]).poly_sigmoid(1.2, 4).mean(axis=(-4,-3,-2))

    def __repr__(self):
        return f"RingConv(input_size={self._weights[0].shape[0]}, output_size={self._weights[0].shape[3]}, window_size={self.window_size}, padding={self.padding}, stride={self.stride})"


class Sequential(Module):
    def __init__(self, modules: list[Module | Callable[[Any],Tensor]]):
        self.modules = modules

    @property
    def weights(self):
        return [w for m in self.modules if isinstance(m, Module) for w in m.weights]

    def forward(self, x: Tensor) -> Tensor:
        for m in self.modules:
            x = m(x)
        return x

    def __repr__(self):
        steps = []
        for m in self.modules:
            if isinstance(m, Module):
                steps.append(m.__repr__())
            elif isinstance(m, Partial):
                steps.append(m.__repr__())
            elif hasattr(m, '__name__') and m.__name__ == '<lambda>' and hasattr(m, '__code__'):
                steps.append(f"lambda {', '.join(m.__code__.co_varnames)}: ...")
            else:
                steps.append(str(m))
        return "\n-> ".join(steps)


class Input(Model):
    # placeholder methods for type annotations
    def ff(self, output_size: int) -> "Input": ...
    def conv(self, output_size: int, window_size: int = 3, padding: int = 1, stride: int = 1) -> "Input": ...
    def apply(self, fn: Callable[[RingTensor], Tensor]) -> "Input": ...
    def flatten(self, *axes: int) -> "Input": ...
    def save(self, path: str): ...
    def load(self, path: str) -> "Sequential": ...
    # also Tensor methods are available

    def __init__(self, shape: tuple[int, ...]):
        self._input_shape = shape
        self._shape = shape
        self._network = Sequential([])

    def __call__(self, x: Tensor) -> Tensor:
        return object.__getattribute__(self, "_network")(x)

    def __getattribute__(self, name: str) -> Any:
        shape = object.__getattribute__(self, "_shape")
        network = object.__getattribute__(self, "_network")

        def add_fn(fn: Callable[[RingTensor], Tensor]) -> "Input":
            output_shape = fn(RingTensor.rand(shape)).shape
            network.modules.append(fn)
            object.__setattr__(self, "_shape", output_shape)
            return self

        match name:
            case "ff":
                return lambda output_size: add_fn(RingFF(shape[-1], output_size))
            case "conv":
                return lambda output_size, window_size=3, padding=1, stride=1: add_fn(RingConv(shape[-1], output_size, window_size, padding, stride))
            case "apply":
                return lambda fn: add_fn(fn)
            case "weights" | "nparams" | "save" | "load":
                return getattr(network, name)
            case _:
                if hasattr(Tensor, name):
                    # add tensor operation to the network
                    return lambda *args, **kwargs: add_fn(Partial(getattr(Tensor, name), *args, **kwargs))
                raise AttributeError(f"Input has no attribute {name}")

    def __repr__(self):
        '''representation including the input and output shapes'''
        input_shape = object.__getattribute__(self, "_input_shape")
        output_shape = object.__getattribute__(self, "_shape")
        network = object.__getattribute__(self, "_network")
        return f"Input(shape={input_shape})\n-> {network!r}\n-> Output(shape={output_shape})"
