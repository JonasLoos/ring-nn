import numpy as np
from typing import Self
import urllib.request
import os


class Tensor:
    dtype = np.int8

    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=self.dtype)
        self._rg = requires_grad
        self._grad = None
        self.reset_grad()
        self._backward = lambda: None
        self._prev = set()

    def reset_grad(self):
        self._grad = np.zeros_like(self.data, dtype=np.float32) if self._rg else None

    def __add__(self, other: Self) -> Self:
        out = Tensor(self.data + other.data, requires_grad=self._rg or other._rg)
        out._prev = {self, other}

        def _backward():
            if self._rg:
                self._grad = self._grad + Tensor._unbroadcast_gradient(out._grad, self.data.shape)
            if other._rg:
                other._grad = other._grad + Tensor._unbroadcast_gradient(out._grad, other.data.shape)
        out._backward = _backward
        return out
    
    def sum(self, axis=None):
        out = Tensor(self.data.sum(axis=axis), requires_grad=self._rg)
        out._prev = {self}

        def _backward():
            if self._rg:
                # Create gradient array of same shape as input, filled with the output gradient
                self._grad = self._grad + np.full(self.data.shape, out._grad)
        out._backward = _backward
        return out
    
    def mean(self, axis=None):
        out = Tensor(self.data.mean(axis=axis), requires_grad=self._rg)
        out._prev = {self}

        def _backward():
            if self._rg:
                # If out.data.size is 0, it means the output of np.mean was an empty array.
                if out.data.size == 0:
                    return

                num_elements_averaged = self.data.size // out.data.size
                if num_elements_averaged > 0:
                    # Scale the output gradient by 1/N.
                    scaled_grad_values = out._grad.astype(np.float32) / num_elements_averaged
                    # Create the full gradient contribution by broadcasting
                    grad_contribution_float = np.full(self.data.shape, scaled_grad_values, dtype=np.float32)
                    self._grad = self._grad + grad_contribution_float
        out._backward = _backward
        return out
    
    
    def __neg__(self) -> Self:
        return self + Tensor(np.full_like(self.data, 128))
    
    def sin(self) -> Self:
        # TODO: use ((sin(x/256*pi-pi/2) + 1) * -(x>0) * 128).clip(-128, 127), i.e. double sigmoid-like activation
        out = Tensor((np.sin(self.data.astype(np.float32) / 256 * np.pi) * 128).clip(-128, 127), requires_grad=self._rg)
        out._prev = {self}

        def _backward():
            if self._rg:
                # Calculate the local gradient, keeping it as float
                local_grad = (np.cos(self.data.astype(np.float32) / 256 * np.pi) * 128 / 256 * np.pi) # Derivative of sin(x * C1) * C2 is cos(x * C1) * C1 * C2
                # The original forward pass was (np.sin(self.data.astype(np.float32) / 256 * np.pi) * 128)
                # So C1 = np.pi / 256, C2 = 128. Derivative: cos(self.data * C1) * C1 * C2
                self._grad = self._grad + out._grad * local_grad
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
        self._grad = np.ones_like(self.data, dtype=np.float32)
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self._grad})"

    @classmethod
    def rand(cls, shape, requires_grad=False):
        return cls(np.random.randint(-128, 127, size=shape, dtype=cls.dtype), requires_grad=requires_grad)

    @property
    def shape(self):
        return self.data.shape

    def reshape(self, shape):
        out = Tensor(self.data.reshape(shape), requires_grad=self._rg)
        out._prev = {self}

        def _backward():
            if self._rg:
                # Reshape the gradient back to the original shape
                self._grad = self._grad + out._grad.reshape(self.data.shape)
        out._backward = _backward
        return out

    @staticmethod
    def _unbroadcast_gradient(grad_output, original_input_shape):
        """
        Sums grad_output along axes to revert broadcasting, so it matches original_input_shape.
        """
        processed_grad = np.array(grad_output, copy=False) # Operate on a view
        shape_out = list(processed_grad.shape)
        shape_in = list(original_input_shape)

        # 1. Sum over leading axes in grad_output not present in original_input_shape
        delta_dims = len(shape_out) - len(shape_in)
        if delta_dims > 0:
            processed_grad = np.sum(processed_grad, axis=tuple(range(delta_dims)))
            shape_out = list(processed_grad.shape) # Update shape_out

        # 2. Sum over axes that were 1 in original_input_shape but >1 in processed_grad's shape
        axes_to_sum_expansion = []
        len_shape_out = len(shape_out)
        len_shape_in = len(shape_in)
        for i in range(len_shape_out): # Iterate based on current processed_grad dimensions
            out_axis_idx = len_shape_out - 1 - i # Current axis in processed_grad (from right)
            in_axis_idx = len_shape_in - 1 - i   # Corresponding axis in original_input_shape (from right)

            if in_axis_idx >= 0: # Check if original_input_shape has this dimension
                if shape_in[in_axis_idx] == 1 and shape_out[out_axis_idx] > 1:
                    axes_to_sum_expansion.append(out_axis_idx)
        
        if axes_to_sum_expansion:
            processed_grad = np.sum(processed_grad, axis=tuple(axes_to_sum_expansion), keepdims=True)
        
        return np.reshape(processed_grad, original_input_shape)


def test_overflow():
    a = Tensor([100], requires_grad=True)
    b = Tensor([100], requires_grad=True)

    c = a + b
    c.backward()

    print(f"{c = }")
    print(f"{a._grad = }")
    print(f"{b._grad = }")


def test_int8_overflow_gradients():
    # Test case 1: Simple addition with overflow
    a = Tensor([100], requires_grad=True)  # int8: 100
    b = Tensor([100], requires_grad=True)  # int8: 100
    c = a + b  # Should overflow to -56 (100 + 100 = 200, which wraps to -56 in int8)
    c.backward()
    print("\nTest 1: Simple addition with overflow")
    print(f"c = {c}")  # Should be -56
    print(f"a._grad = {a._grad}")  # Should be 1
    print(f"b._grad = {b._grad}")  # Should be 1

    # Test case 2: Multiplication with overflow
    x = Tensor([10], requires_grad=True)  # int8: 10
    y = Tensor([20], requires_grad=True)  # int8: 20
    z = x * y  # Should be 200, which wraps to -56 in int8
    z.backward()
    print("\nTest 2: Multiplication with overflow")
    print(f"z = {z}")  # Should be -56
    print(f"x._grad = {x._grad}")  # Should be 20
    print(f"y._grad = {y._grad}")  # Should be 10

    # Test case 3: Chain of operations with overflow
    p = Tensor([50], requires_grad=True)  # int8: 50
    q = Tensor([3], requires_grad=True)   # int8: 3
    r = p * q  # 150, which wraps to -106 in int8
    s = r + Tensor([100], requires_grad=True)  # -106 + 100 = -6
    s.backward()
    print("\nTest 3: Chain of operations with overflow")
    print(f"s = {s}")  # Should be -6
    print(f"p._grad = {p._grad}")  # Should be 3
    print(f"q._grad = {q._grad}")  # Should be 50

    # Test case 4: Power operations with overflow
    base = Tensor([5], requires_grad=True)    # int8: 5
    power = 3
    result = base ** power  # 5^3 = 125, which is within int8 range
    result.backward()
    print("\nTest 4: Power operation within range")
    print(f"result = {result}")  # Should be 125
    print(f"base._grad = {base._grad}")  # Should be 75 (3 * 5^2)

    # Test case 5: Power operations with overflow
    base2 = Tensor([10], requires_grad=True)  # int8: 10
    power2 = 3
    result2 = base2 ** power2  # 10^3 = 1000, which wraps to -24 in int8
    result2.backward()
    print("\nTest 5: Power operation with overflow")
    print(f"result2 = {result2}")  # Should be -24
    print(f"base2._grad = {base2._grad}")  # Should be 300 (3 * 10^2)

    # Test case 6: Chain of power operations
    base3 = Tensor([3], requires_grad=True)   # int8: 3
    result3 = (base3 ** 2) ** 2  # (3^2)^2 = 81, which is within int8 range
    result3.backward()
    print("\nTest 6: Chain of power operations")
    print(f"result3 = {result3}")  # Should be 81
    print(f"base3._grad = {base3._grad}")  # Should be 108 (4 * 3^3)


def test_sum():
    # Test case 1: Basic sum operation
    x = Tensor([[1, 2], [3, 4]], requires_grad=True)
    y = x.sum()  # Should be 10
    y.backward()
    print("\nTest 1: Basic sum operation")
    print(f"x = {x.data}")
    print(f"y = {y.data}")  # Should be 10
    print(f"x._grad = {x._grad}")  # Should be [[1, 1], [1, 1]]

    # Test case 2: Sum along axis 0
    a = Tensor([[1, 2], [3, 4]], requires_grad=True)
    b = a.sum(axis=0)  # Should be [4, 6]
    b.backward()
    print("\nTest 2: Sum along axis 0")
    print(f"a = {a.data}")
    print(f"b = {b.data}")  # Should be [4, 6]
    print(f"a._grad = {a._grad}")  # Should be [[1, 1], [1, 1]]

    # Test case 3: Sum along axis 1
    p = Tensor([[1, 2], [3, 4]], requires_grad=True)
    q = p.sum(axis=1)  # Should be [3, 7]
    q.backward()
    print("\nTest 3: Sum along axis 1")
    print(f"p = {p.data}")
    print(f"q = {q.data}")  # Should be [3, 7]
    print(f"p._grad = {p._grad}")  # Should be [[1, 1], [1, 1]]


def test_reshape():
    x = Tensor([[1, 2], [3, 4]], requires_grad=True)
    y = x.reshape((4, 1))
    y.backward()
    print(f"x = {x.data}")  # Should be [[1, 2], [3, 4]]
    print(f"y = {y.data}")  # Should be [[1], [2], [3], [4]]
    print(f"x._grad = {x._grad}")  # Should be [[1, 1], [1, 1]]


def ring_nn():
    # create a ring network to solve mnist
    weights = [
        Tensor.rand((784, 100), requires_grad=True),
        Tensor.rand((100, 100), requires_grad=True),
        Tensor.rand((100, 10), requires_grad=True),
    ]

    def nn(x):
        for w in weights:
            x = (x + w).sin().mean(axis=0).reshape((-1, 1))
        return x

    # Download MNIST dataset if not already present
    mnist_url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
    mnist_path = 'mnist.npz'
    
    if not os.path.exists(mnist_path):
        print("Downloading MNIST dataset...")
        urllib.request.urlretrieve(mnist_url, mnist_path)

    def convert_x(data):
        return [Tensor(x).reshape((784, 1)) for x in data]
    
    def convert_y(data):
        result = []
        for y in data:
            tmp = np.zeros((10, 1))
            tmp[y, 0] = -128
            result.append(Tensor(tmp))
        return result

    data = np.load(mnist_path)
    x_train, y_train = convert_x(data['x_train']), convert_y(data['y_train'])
    x_test, y_test = convert_x(data['x_test']), convert_y(data['y_test'])

    for epoch in range(10):
        print("-" * 100)
        print(f"Epoch {epoch}")
        print("-" * 100)
        loss = Tensor([0])
        for i, (x, y) in enumerate(zip(x_train, y_train)):
            loss = (nn(x) + y).sum()
            if i % 500 == 0:
                avg_grandient_change = 0
                loss.backward()
                for w in weights:
                    w.data = w.data - w._grad / 100
                    # print(w._grad)
                    print(w._grad.sum() / 100 / np.prod(w.shape))
                    avg_grandient_change += w._grad.sum() / 100 / np.prod(w.shape)
                    w.reset_grad()
                avg_grandient_change /= len(weights)
                print(f"[{i:05}/{len(x_train)}]: loss: {loss.data.item():5} | Avg. gradient change: {avg_grandient_change}")
                loss = Tensor([0])

        # Test on validation set
        test_loss = Tensor([0])
        for x, y in zip(x_test, y_test):
            test_loss = test_loss + (nn(x) + y).sum()
        print(f"\nTest loss: {test_loss.data.item()}")


if __name__ == '__main__':
    # test_overflow()
    # test_int8_overflow_gradients()
    # test_sum()
    # test_reshape()
    ring_nn()
