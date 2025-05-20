import numpy as np
from typing import Self


class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set()

    def __add__(self, other: Self) -> Self:
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad if self.grad is not None else out.grad
            if other.requires_grad:
                other.grad = other.grad + out.grad if other.grad is not None else out.grad
        out._backward = _backward
        return out

    def __sub__(self, other: Self) -> Self:
        return self + (other * -1)

    def __mul__(self, other: Self) -> Self:
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + other.data * out.grad if self.grad is not None else other.data * out.grad
            if other.requires_grad:
                other.grad = other.grad + self.data * out.grad if other.grad is not None else self.data * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other: Self) -> Self:
        return self * other**-1

    def __pow__(self, power):
        out = Tensor(self.data**power, requires_grad=self.requires_grad)
        out._prev = {self}

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + (power * self.data**(power - 1) * out.grad) if self.grad is not None else (power * self.data**(power - 1) * out.grad)
        out._backward = _backward
        return out

    def __matmul__(self, other: Self) -> Self:
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + (out.grad @ other.data.T) if self.grad is not None else (out.grad @ other.data.T)
            if other.requires_grad:
                other.grad = other.grad + (self.data.T @ out.grad) if other.grad is not None else (self.data.T @ out.grad)
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad)
        out._prev = {self}

        def _backward():
            if self.requires_grad:
                grad = out.grad * (self.data > 0)
                self.grad = self.grad + grad if self.grad is not None else grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)
        build_topo(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"



if __name__ == '__main__':
    # Test: Simple MLP forward and backward
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    w = Tensor([[2.0], [3.0]], requires_grad=True)
    b = Tensor([1.0], requires_grad=True)

    z = x @ w + b
    out = z.relu()
    loss = out ** 2
    loss.backward()

    print(f"{loss = }")
    print("x.grad:\n", x.grad)
    print("w.grad:\n", w.grad)
    print("b.grad:\n", b.grad)
