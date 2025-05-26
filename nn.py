from tensor import Tensor, RingTensor
import pickle
from typing import Callable


class Module:
    _weights: list[Tensor] = []

    @property
    def weights(self):
        all_weights = self._weights
        for m in self.__dict__.values():
            if isinstance(m, Module):
                all_weights += m.weights
        return all_weights
        
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.weights, f)

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            weights = pickle.load(f)
        nn = cls([w.shape[0] for w in weights] + [weights[-1].shape[1]])
        nn.weights = weights
        return nn


class RingFF(Module):
    def __init__(self, input_size: int, output_size: int):
        self._weights = [RingTensor.rand((input_size, output_size), requires_grad=True)]

    def __call__(self, x: Tensor):
        return (x.unsqueeze(-1) - self._weights[0]).poly_sigmoid(1.2, 4).mean(axis=-2)


class RingConv(Module):
    def __init__(self, input_size: int, output_size: int, window_size: int = 3, padding: int = 1, stride: int = 1):
        self.window_size = window_size
        self.padding = padding
        self.stride = stride
        self._weights = [RingTensor.rand((input_size, window_size, window_size, output_size), requires_grad=True)]

    def __call__(self, x: Tensor):
        return (x.sliding_window_2d(self.window_size, self.padding, self.stride).unsqueeze(-1) - self._weights[0]).poly_sigmoid(1.2, 4).mean(axis=(-4,-3,-2))


class Sequential(Module):
    def __init__(self, modules: list[Module | Callable]):
        self.modules = modules

    def __call__(self, x: Tensor):
        for m in self.modules:
            x = m(x)
        return x
