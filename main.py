import numpy as np
from typing import Self
import urllib.request
import os


class Tensor:
    dtype = np.int8

    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=self.dtype)
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
    
    def sum(self, axis=None):
        out = Tensor(self.data.sum(axis=axis), requires_grad=self.requires_grad)
        out._prev = {self}

        def _backward():
            if self.requires_grad:
                # Create gradient array of same shape as input, filled with the output gradient
                expanded_grad = np.full(self.data.shape, out.grad)
                self.grad = self.grad + expanded_grad if self.grad is not None else expanded_grad
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

    @classmethod
    def rand(cls, shape, requires_grad=False):
        return cls(np.random.randint(-128, 127, size=shape, dtype=cls.dtype), requires_grad=requires_grad)

    @property
    def shape(self):
        return self.data.shape

    def reshape(self, shape):
        out = Tensor(self.data.reshape(shape), requires_grad=self.requires_grad)
        out._prev = {self}

        def _backward():
            if self.requires_grad:
                # Reshape the gradient back to the original shape
                grad = out.grad.reshape(self.data.shape)
                self.grad = self.grad + grad if self.grad is not None else grad
        out._backward = _backward
        return out


def test_mlp():
    # Test: Simple MLP forward and backward
    x = Tensor([[1, 2]], requires_grad=True)
    w = Tensor([[2], [3]], requires_grad=True)
    b = Tensor([1], requires_grad=True)

    z = x @ w + b
    out = z.relu()
    loss = out ** 2
    loss.backward()

    print(f"{loss = }")
    print("x.grad:\n", x.grad)
    print("w.grad:\n", w.grad)
    print("b.grad:\n", b.grad)


def test_overflow():
    a = Tensor([100], requires_grad=True)
    b = Tensor([100], requires_grad=True)

    c = a + b
    c.backward()

    print(f"{c = }")
    print(f"{a.grad = }")
    print(f"{b.grad = }")


def test_int8_overflow_gradients():
    # Test case 1: Simple addition with overflow
    a = Tensor([100], requires_grad=True)  # int8: 100
    b = Tensor([100], requires_grad=True)  # int8: 100
    c = a + b  # Should overflow to -56 (100 + 100 = 200, which wraps to -56 in int8)
    c.backward()
    print("\nTest 1: Simple addition with overflow")
    print(f"c = {c}")  # Should be -56
    print(f"a.grad = {a.grad}")  # Should be 1
    print(f"b.grad = {b.grad}")  # Should be 1

    # Test case 2: Multiplication with overflow
    x = Tensor([10], requires_grad=True)  # int8: 10
    y = Tensor([20], requires_grad=True)  # int8: 20
    z = x * y  # Should be 200, which wraps to -56 in int8
    z.backward()
    print("\nTest 2: Multiplication with overflow")
    print(f"z = {z}")  # Should be -56
    print(f"x.grad = {x.grad}")  # Should be 20
    print(f"y.grad = {y.grad}")  # Should be 10

    # Test case 3: Chain of operations with overflow
    p = Tensor([50], requires_grad=True)  # int8: 50
    q = Tensor([3], requires_grad=True)   # int8: 3
    r = p * q  # 150, which wraps to -106 in int8
    s = r + Tensor([100], requires_grad=True)  # -106 + 100 = -6
    s.backward()
    print("\nTest 3: Chain of operations with overflow")
    print(f"s = {s}")  # Should be -6
    print(f"p.grad = {p.grad}")  # Should be 3
    print(f"q.grad = {q.grad}")  # Should be 50

    # Test case 4: Power operations with overflow
    base = Tensor([5], requires_grad=True)    # int8: 5
    power = 3
    result = base ** power  # 5^3 = 125, which is within int8 range
    result.backward()
    print("\nTest 4: Power operation within range")
    print(f"result = {result}")  # Should be 125
    print(f"base.grad = {base.grad}")  # Should be 75 (3 * 5^2)

    # Test case 5: Power operations with overflow
    base2 = Tensor([10], requires_grad=True)  # int8: 10
    power2 = 3
    result2 = base2 ** power2  # 10^3 = 1000, which wraps to -24 in int8
    result2.backward()
    print("\nTest 5: Power operation with overflow")
    print(f"result2 = {result2}")  # Should be -24
    print(f"base2.grad = {base2.grad}")  # Should be 300 (3 * 10^2)

    # Test case 6: Chain of power operations
    base3 = Tensor([3], requires_grad=True)   # int8: 3
    result3 = (base3 ** 2) ** 2  # (3^2)^2 = 81, which is within int8 range
    result3.backward()
    print("\nTest 6: Chain of power operations")
    print(f"result3 = {result3}")  # Should be 81
    print(f"base3.grad = {base3.grad}")  # Should be 108 (4 * 3^3)


def test_sum():
    # Test case 1: Basic sum operation
    x = Tensor([[1, 2], [3, 4]], requires_grad=True)
    y = x.sum()  # Should be 10
    y.backward()
    print("\nTest 1: Basic sum operation")
    print(f"x = {x.data}")
    print(f"y = {y.data}")  # Should be 10
    print(f"x.grad = {x.grad}")  # Should be [[1, 1], [1, 1]]

    # Test case 2: Sum along axis 0
    a = Tensor([[1, 2], [3, 4]], requires_grad=True)
    b = a.sum(axis=0)  # Should be [4, 6]
    b.backward()
    print("\nTest 2: Sum along axis 0")
    print(f"a = {a.data}")
    print(f"b = {b.data}")  # Should be [4, 6]
    print(f"a.grad = {a.grad}")  # Should be [[1, 1], [1, 1]]

    # Test case 3: Sum along axis 1
    p = Tensor([[1, 2], [3, 4]], requires_grad=True)
    q = p.sum(axis=1)  # Should be [3, 7]
    q.backward()
    print("\nTest 3: Sum along axis 1")
    print(f"p = {p.data}")
    print(f"q = {q.data}")  # Should be [3, 7]
    print(f"p.grad = {p.grad}")  # Should be [[1, 1], [1, 1]]


def test_reshape():
    x = Tensor([[1, 2], [3, 4]], requires_grad=True)
    y = x.reshape((4, 1))
    y.backward()
    print(f"x = {x.data}")  # Should be [[1, 2], [3, 4]]
    print(f"y = {y.data}")  # Should be [[1], [2], [3], [4]]
    print(f"x.grad = {x.grad}")  # Should be [[1, 1], [1, 1]]


def ring_nn():
    # create a ring network to solve mnist
    weights = [
        Tensor.rand((784, 100), requires_grad=True),
        Tensor.rand((100, 100), requires_grad=True),
        Tensor.rand((100, 10), requires_grad=True),
    ]

    def nn(x):
        for w in weights:
            x = (x + w).sum(axis=0).reshape((-1, 1))
        return x

    # Download MNIST dataset if not already present
    mnist_url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
    mnist_path = 'mnist.npz'
    
    if not os.path.exists(mnist_path):
        print("Downloading MNIST dataset...")
        urllib.request.urlretrieve(mnist_url, mnist_path)
    
    data = np.load(mnist_path)
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']

    for epoch in range(10):
        print("-" * 100)
        print(f"Epoch {epoch}")
        print("-" * 100)
        loss = Tensor([0])
        for i, (x, y) in enumerate(zip(x_train, y_train)):
            x = Tensor(x//2).reshape((784, 1))
            tmp = np.zeros((10, 1))
            tmp[y, 0] = 1
            y = Tensor(tmp)
            loss = (nn(x) + y).sum()
            if i % 100 == 0:
                loss.backward()
                print(f"loss: {loss}")
                for w in weights:
                    w.data = w.data - w.grad // 100
                    w.grad = None
                loss = Tensor([0])

    # Test on validation set
    test_loss = Tensor([0])
    for x, y in zip(x_test, y_test):
        x = Tensor(x//2).reshape((784, 1))
        tmp = np.zeros((10, 1))
        tmp[y, 0] = 1
        y = Tensor(tmp)
        test_loss = test_loss + (nn(x) + y).sum()
    print(f"\nTest loss: {test_loss}")


if __name__ == '__main__':
    # test_mlp()
    # test_overflow()
    # test_int8_overflow_gradients()
    # test_sum()
    # test_reshape()
    ring_nn()
