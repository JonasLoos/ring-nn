from tensor import Tensor, RingTensor
import pickle
from typing import Callable


class Module:
    _weights: list[Tensor] | None = None

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

    def __call__(self, x: Tensor) -> Tensor:
        return (x.unsqueeze(-1) - self._weights[0]).poly_sigmoid(1.2, 4).mean(axis=-2)


class RingConv(Module):
    def __init__(self, input_size: int, output_size: int, window_size: int = 3, padding: int = 1, stride: int = 1):
        self.window_size = window_size
        self.padding = padding
        self.stride = stride
        self._weights = [RingTensor.rand((input_size, window_size, window_size, output_size), requires_grad=True)]

    def __call__(self, x: Tensor) -> Tensor:
        return (x.sliding_window_2d(self.window_size, self.padding, self.stride).unsqueeze(-1) - self._weights[0]).poly_sigmoid(1.2, 4).mean(axis=(-4,-3,-2))


class Sequential(Module):
    def __init__(self, modules: list[Module | Callable[[Tensor],Tensor]]):
        self.modules = modules

    @property
    def weights(self):
        return [w for m in self.modules if isinstance(m, Module) for w in m.weights]

    def __call__(self, x: Tensor) -> Tensor:
        for m in self.modules:
            x = m(x)
        return x
